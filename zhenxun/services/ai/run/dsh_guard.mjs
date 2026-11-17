import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import { fileURLToPath } from 'node:url'

export const name = 'zhenxun-desktop-guard'
export const inject = ['tools', 'agentDefaultModel']

const TOOL_NAMES = {
  chat: [],
  agent: ['ask_user_question', 'zx_list', 'zx_read', 'zx_write']
}
const RETRY_DELAYS = [2000, 5000, 10000, 20000]
const RETRY_CODES = new Set(['EMPTY_RESPONSE', 'RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT'])

/** 等待下一次模型尝试，取消立即清除定时器。 */
function retryDelay(ms, signal) {
  signal.throwIfAborted()
  return new Promise((resolve, reject) => {
    const aborted = () => { clearTimeout(timer); reject(signal.reason) }
    const timer = setTimeout(() => { signal.removeEventListener('abort', aborted); resolve() }, ms)
    signal.addEventListener('abort', aborted, { once: true })
    if (signal.aborted) aborted()
  })
}

/** 对同一模型请求最多执行四次可取消重试，不重放工具或用户输入。 */
export async function retryModelRequest(payload, run, expires, check) {
  const { agent, turn, step, provider, failure, signal } = payload
  if (run.uncertain || !RETRY_CODES.has(failure.code)) return undefined
  const key = String(turn) + ':' + String(step)
  const previous = run.retries.get(key) || { count: 0, delay: 0 }
  if (previous.count >= RETRY_DELAYS.length) return undefined
  const retry = previous.count + 1
  const hint = Number(failure.providerRetryAfterMs)
  const delayMs = Math.max(RETRY_DELAYS[previous.count], previous.delay * 2, Number.isFinite(hint) && hint > 0 ? hint : 0)
  if (Date.now() + delayMs >= expires * 1000) return undefined
  run.retries.set(key, { count: retry, delay: delayMs })
  const retryId = randomUUID()
  agent.session.append('llm/retry', { retryId, turn, step, provider, mode: 'normal', policyKey: 'zhenxun-five-attempts', retry, maxRetries: 4, delayMs, failure })
  await retryDelay(delayMs, signal)
  await check()
  agent.session.append('llm/retry-started', { retryId, turn, step, retry })
  return { kind: 'retry' }
}

/** 原生桌面会话只记录 QQ 请求的重试范围，不改变工具、预设或模型。 */
async function installForwardingRetry(ctx, config, revision) {
  const owned = new WeakMap()
  let readWarning = false
  const marker = path.join(config.root, 'forwarding-ready.json')
  await fs.mkdir(config.root, { recursive: true })
  await fs.writeFile(marker, JSON.stringify({ pid: process.pid, revision }))
  ctx.effect(() => async () => {
    try {
      const value = JSON.parse(await fs.readFile(marker, 'utf8'))
      if (value.pid === process.pid && value.revision === revision) await fs.unlink(marker)
    } catch (error) { if (error.code !== 'ENOENT') ctx.logger.warn('Zhenxun forwarding marker cleanup failed') }
  })
  async function leaseFor(agent) {
    if (!/^[a-zA-Z0-9_-]{1,128}$/.test(agent.id)) return null
    try {
      const lease = JSON.parse(await fs.readFile(path.join(config.root, 'leases', agent.id + '.json'), 'utf8'))
      readWarning = false
      return lease.native === true && lease.sessionId === agent.id ? lease : null
    } catch (error) {
      if (error.code !== 'ENOENT' && !readWarning) {
        readWarning = true
        ctx.logger.warn('Zhenxun forwarding receipt unavailable; leaving unrelated desktop execution unchanged')
      }
      return null
    }
  }
  ctx.on('agent/pre-step', async (payload, next) => {
    const current = owned.get(payload.agent)
    if (current && current.turn !== payload.turn) owned.delete(payload.agent)
    const lease = await leaseFor(payload.agent)
    if (lease && payload.messages.some(message => message.source?.rpcId === lease.requestId)) {
      owned.set(payload.agent, { id: lease.runId, turn: payload.turn, retries: new Map(), uncertain: false })
    }
    return next()
  }, { prepend: true })
  ctx.on('agent/request-error', async (payload, next) => {
    const run = owned.get(payload.agent)
    if (!run || run.turn !== payload.turn) return next()
    const lease = await leaseFor(payload.agent)
    if (!lease || lease.runId !== run.id || lease.expires <= Date.now() / 1000) return undefined
    const check = async () => {
      const url = new URL(lease.authorize)
      if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.pathname !== '/zhenxun/api/ai/dsh/authorize' || url.search || url.username || url.password) throw new Error('Invalid forwarding receipt target')
      const response = await fetch(url, { method: 'POST', redirect: 'error', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + lease.token }, body: JSON.stringify({ sessionId: payload.agent.id, runId: run.id, mode: 'agent', scope: 'retry', revision, tools: [] }), signal: AbortSignal.any([payload.signal, AbortSignal.timeout(5000)]) })
      if (!response.ok || !(await response.json()).allowed) throw new Error('Forwarding retry no longer owned')
    }
    await check()
    return retryModelRequest(payload, run, lease.expires, check)
  }, { prepend: true })
}

/** 为真寻预设绑定当前运行、逐次授权及有限工作目录工具。 */
export async function apply(ctx, config) {
  if (!path.isAbsolute(config.root)) throw new Error('Invalid Zhenxun integration root')
  const revision = createHash('sha256').update(await fs.readFile(fileURLToPath(import.meta.url))).digest('hex')
  if (config.mode === 'bridge') return installForwardingRetry(ctx, config, revision)
  if (!TOOL_NAMES[config.mode]) throw new Error('Invalid Zhenxun preset')
  const approved = new WeakSet()
  const runs = new WeakMap()

  async function authorize(agent, tool, signal) {
    if (!agent || !/^zx-[a-f0-9]{32}$/.test(agent.id)) throw new Error('Unowned Zhenxun session')
    const lease = JSON.parse(await fs.readFile(path.join(config.root, 'leases', agent.id + '.json'), 'utf8'))
    if (lease.sessionId !== agent.id || lease.mode !== config.mode || lease.expires <= Date.now() / 1000 || path.resolve(lease.cwd) !== path.resolve(agent.session.header.cwd || '')) throw new Error('Expired or mismatched Zhenxun run')
    const url = new URL(lease.authorize)
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.pathname !== '/zhenxun/api/ai/dsh/authorize' || url.search || url.username || url.password) throw new Error('Invalid authorization target')
    const names = ctx.tools.definitions(agent).map(t => t.name).sort()
    if (JSON.stringify(names) !== JSON.stringify([...TOOL_NAMES[config.mode]].sort())) throw new Error('Unexpected tools in Zhenxun preset: ' + names.join(', '))
    let run = runs.get(agent)
    if (!run || run.id !== lease.runId) {
      run = { id: lease.runId, model: { ...(lease.modelSelection || ctx.agentDefaultModel.currentSelection()) }, writes: new Map(), retries: new Map(), uncertain: false }
      if (!['deepseek', 'deepseek-account'].includes(run.model.provider)) throw new Error('Unverified Zhenxun model transport')
      runs.set(agent, run)
      if (!lease.modelSelection) agent.model.select(run.model)
    }
    try {
      const response = await fetch(url, {
        method: 'POST', redirect: 'error',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + lease.token },
        body: JSON.stringify({ sessionId: agent.id, runId: lease.runId, mode: config.mode, revision, tools: names, tool, model: run.model }),
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(5000)]) : AbortSignal.timeout(5000)
      })
      if (!response.ok || (await response.json()).allowed !== true) throw new Error('Run authorization denied')
    } catch (error) {
      agent.cancel({ kind: 'user' }, { keepInbox: true })
      throw new Error('Zhenxun authorization unavailable', { cause: error })
    }
    return { lease, run }
  }

  ctx.on('agent/created', async ({ agent, signal }) => {
    agent.ctx.tools.restrict({ allow: TOOL_NAMES[config.mode] })
    await authorize(agent, undefined, signal)
  })
  ctx.on('agent/pre-step', async (payload, next) => {
    const { run } = await authorize(payload.agent, undefined, payload.signal)
    if (payload.messages.some(message => message.content?.some(block => block.type === 'image'))) throw new Error('Images require provider recovery that is not allowed in Zhenxun runs')
    if (payload.step > 32 || run.uncertain) throw new Error('Zhenxun run budget or uncertain side effect')
    Object.assign(payload.modelSelection, run.model)
    return next()
  }, { prepend: true })
  ctx.on('agent/request-error', async (payload) => {
    const { run, lease } = await authorize(payload.agent, undefined, payload.signal)
    return retryModelRequest(payload, run, lease.expires, () => authorize(payload.agent, undefined, payload.signal))
  }, { prepend: true })
  ctx.on('tools/pre-execute', async (exec, next) => {
    await authorize(exec.agent, exec.name, exec.signal)
    approved.add(exec)
    return next()
  }, { prepend: true })
  ctx.tools.guard(exec => {
    if (!approved.has(exec) || !TOOL_NAMES[config.mode].includes(exec.name)) return 'Zhenxun tool authorization required'
  })
  ctx.on('tools/execute', async (exec, next) => {
    await authorize(exec.agent, exec.name, exec.signal)
    return next()
  }, { prepend: true })

  async function inside(root, relative, writing = false) {
    if (typeof relative !== 'string' || path.isAbsolute(relative) || relative.includes(':') || relative.split(/[\\/]/).some(p => p === '..' || p.startsWith('.'))) throw new Error('Only visible relative workspace paths are allowed')
    const base = await fs.realpath(root)
    const target = path.resolve(base, relative || '.')
    if (target !== base && !target.startsWith(base + path.sep)) throw new Error('Path outside workspace')
    const parts = path.relative(base, target).split(path.sep).filter(Boolean)
    let cursor = base
    for (let i = 0; i < parts.length; i++) {
      cursor = path.join(cursor, parts[i])
      try {
        if ((await fs.lstat(cursor)).isSymbolicLink()) throw new Error('Workspace links are not allowed')
      } catch (error) {
        if (error.code !== 'ENOENT' || !writing || i !== parts.length - 1) throw error
      }
    }
    return target
  }

  function register(name, description, properties, required, execute) {
    ctx.tools.register({
      name, description,
      parameters: { type: 'object', properties, required, additionalProperties: false },
      output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
      timeoutMs: 10000,
      execute
    })
  }
  if (config.mode === 'agent') {
    const file = { type: 'string', description: 'Relative path in this owner workspace' }
    register('zx_list', 'List files in your isolated workspace. No host or other session access.', { path: file }, ['path'], async (args, exec) => {
      const { lease } = await authorize(exec.agent, exec.name, exec.signal)
      const target = await inside(lease.cwd, args.path)
      const entries = await fs.readdir(target, { withFileTypes: true })
      exec.signal.throwIfAborted()
      return entries.filter(e => !e.name.startsWith('.') && !e.isSymbolicLink()).slice(0, 200).map(e => e.name + (e.isDirectory() ? '/' : '')).join('\n')
    })
    register('zx_read', 'Read a text file in your isolated workspace (maximum 128 KiB).', { path: file }, ['path'], async (args, exec) => {
      const { lease } = await authorize(exec.agent, exec.name, exec.signal)
      const target = await inside(lease.cwd, args.path)
      if ((await fs.stat(target)).size > 131072) throw new Error('File too large')
      const value = await fs.readFile(target, { encoding: 'utf8', signal: exec.signal })
      exec.signal.throwIfAborted()
      return value
    })
    register('zx_write', 'Write a text file in your isolated workspace (maximum 128 KiB). Never writes outside it.', { path: file, text: { type: 'string' } }, ['path', 'text'], async (args, exec) => {
      const { lease, run } = await authorize(exec.agent, exec.name, exec.signal)
      if (typeof args.text !== 'string' || Buffer.byteLength(args.text) > 131072) throw new Error('Invalid text size')
      const target = await inside(lease.cwd, args.path, true)
      const operation = createHash('sha256').update(JSON.stringify([target, args.text])).digest('hex')
      if (run.uncertain) throw new Error('Previous write result unknown; cannot replay')
      if (run.writes.has(operation)) return run.writes.get(operation)
      const temporary = path.join(path.dirname(target), '.zx-' + randomUUID())
      run.uncertain = true
      try {
        await fs.writeFile(temporary, args.text, { flag: 'wx', signal: exec.signal })
        await authorize(exec.agent, exec.name, exec.signal)
        exec.signal.throwIfAborted()
        await fs.rename(temporary, target)
        const result = 'Saved ' + args.path
        run.writes.set(operation, result)
        run.uncertain = false
        return result
      } catch (error) {
        exec.agent.cancel({ kind: 'user' }, { keepInbox: true })
        throw error
      } finally {
        await fs.unlink(temporary).catch(() => {})
      }
    })
  }
}
