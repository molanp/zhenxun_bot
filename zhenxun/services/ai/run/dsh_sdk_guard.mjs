import http from 'node:http'
import fs from 'node:fs/promises'
import path from 'node:path'
import { timingSafeEqual } from 'node:crypto'
import { retryModelRequest } from './dsh_guard.mjs'

export const name = 'zhenxun-sdk-control'
export const inject = ['agents', 'tools', 'sessionPersistence', 'workingDirectory', 'llm']

function selection(events) {
  let result
  for (const event of events) {
    if (event.type === 'model/selection') result = { ...event.data }
    if (event.type === 'request/header') {
      const header = event.data.header
      const config = header?.config
      if (config?.provider && config?.model) {
        result = { provider: config.provider, model: config.model }
        if (header.adapterDefaults?.reasoningEffort !== true && config.reasoningEffort) result.reasoningEffort = config.reasoningEffort
      }
    }
  }
  return result
}

/** 为独立 SDK 提供仅本机控制，普通聊天在执行端禁止宿主工具。 */
export async function apply(ctx, config) {
  if (!['chat', 'agent'].includes(config.mode) || !path.isAbsolute(config.root)) throw new Error('Invalid SDK integration configuration')
  const secret = process.env.ZHENXUN_SDK_CONTROL_TOKEN
  if (!secret || secret.length < 32) throw new Error('Missing SDK control credential')
  const restored = new WeakMap()
  const retries = new WeakMap()
  const expected = new Map()
  const endpoint = path.join(config.root, 'control.json')
  ctx.on('session/created', session => {
    const previous = selection(session.snapshotEvents())
    if (previous) restored.set(session, previous)
  })
  ctx.on('agent/created', async ({ agent }) => {
    if (agent.session.header.agentPreset && !['zhenxun-chat', 'sdk-native'].includes(agent.session.header.agentPreset)) throw new Error('Cannot preserve this SDK session preset')
    if (typeof agent.cancel !== 'function' || typeof agent.whenIdle !== 'function' || !agent.inbox?.remove || !agent.model?.select) throw new Error('Unsupported SDK Agent contract')
    if (config.mode === 'chat') agent.ctx.tools.restrict({ allow: [] })
    const previous = restored.get(agent.session)
    if (previous) agent.model.select(previous)
    const target = expected.get(agent.id)
    if (target) { agent.model.select(target); expected.delete(agent.id) }
    if (!previous && !target && agent.model.current()) agent.model.select(agent.model.current())
  })
  if (config.mode === 'chat') {
    ctx.tools.guard(() => 'Zhenxun ordinary chat cannot execute tools')
    ctx.on('tools/execute', async () => { throw new Error('Zhenxun ordinary chat cannot execute tools') }, { prepend: true })
  }
  ctx.on('agent/pre-step', async (payload, next) => {
    if (config.mode === 'chat' && ctx.tools.definitions(payload.agent).length) throw new Error('Unexpected chat tool exposure')
    const active = payload.agent.model.current()
    if (active) Object.assign(payload.modelSelection, active)
    return next()
  }, { prepend: true })
  ctx.on('agent/request-error', async payload => {
    let run = retries.get(payload.agent)
    if (!run || run.turn !== payload.turn) {
      run = { turn: payload.turn, retries: new Map(), uncertain: false }
      retries.set(payload.agent, run)
    }
    return retryModelRequest(payload, run, Date.now() / 1000 + 1800, async () => { payload.signal.throwIfAborted() })
  }, { prepend: true })

  async function describe(id) {
    const agent = ctx.agents.get(id)
    const stored = await ctx.sessionPersistence.stat(id)
    if (!stored) throw new Error('Session not found')
    let model = agent?.model.current()
    if (!model) {
      const handle = await ctx.sessionPersistence.open(id, 'read')
      try { model = selection((await handle.read()).events) } finally { await handle.close() }
    }
    let cwd = agent && ctx.workingDirectory ? ctx.workingDirectory.get(agent.session) : stored.header.cwd
    if (!agent) {
      const handle = await ctx.sessionPersistence.open(id, 'read')
      try { for (const event of (await handle.read()).events) if (event.type === 'working-directory/change') cwd = event.data.cwd } finally { await handle.close() }
    }
    return { remote_id: id, title: id, cwd, model,
      preset: config.mode === 'chat' ? 'zhenxun-chat' : 'sdk-native', native: config.mode === 'agent',
      running: agent?.status === 'running', loaded: Boolean(agent), subagent: stored.header.origin === 'subagent',
      format: stored.formatStatus || 'current', revision: String(stored.revision), plan: null,
      toolCount: agent ? ctx.tools.definitions(agent).length : null }
  }

  async function dispatch(method, args) {
    if (method === 'capabilities') return { contract: 1, mode: config.mode, pid: process.pid, cancel: true, preserveSelection: true, toolsRestricted: config.mode === 'chat' }
    if (method === 'list') {
      const rows = await ctx.sessionPersistence.list()
      const items = []
      for (const row of rows) {
        if (row.header.origin !== 'subagent' && row.formatStatus !== 'migration-required' && (!row.header.agentPreset || ['zhenxun-chat', 'sdk-native'].includes(row.header.agentPreset))) items.push(await describe(row.header.id))
      }
      return { items }
    }
    if (!/^[a-zA-Z0-9_-]{1,128}$/.test(args.sessionId || '')) throw new Error('Invalid session identity')
    if (method === 'describe') return describe(args.sessionId)
    const agent = ctx.agents.get(args.sessionId)
    if (method === 'prepare') {
      if (!args.model?.provider || !args.model?.model) throw new Error('Missing model selection')
      if (agent?.status === 'running') throw new Error('Session is running')
      await ctx.llm.resolveCallConfig(args.model)
      if (agent) agent.model.select(args.model)
      else expected.set(args.sessionId, { ...args.model })
      return { prepared: true }
    }
    if (!agent) return { running: false, loaded: false, confirmed: false }
    if (method === 'state') return { running: agent.status === 'running', loaded: true, queue: [...agent.inbox.nextTurn, ...agent.inbox.nextStep].map(m => ({ id: m.id })), model: agent.model.current() }
    if (method === 'inspect') {
      const events = agent.session.snapshotEvents()
      const queued = [...agent.inbox.nextTurn, ...agent.inbox.nextStep].some(m => m.id === args.messageId)
      const known = queued || events.some(e => e.type === 'user/message' && (e.data.message || e.data).id === args.messageId)
      return { known, queued, running: agent.status === 'running', loaded: true, foregroundOnly: true }
    }
    if (method === 'cancel') {
      if (!args.messageId) return { confirmed: false }
      const events = agent.session.snapshotEvents()
      const boundary = events.findLastIndex(e => e.type === 'turn/start')
      const queued = [...agent.inbox.nextTurn, ...agent.inbox.nextStep].some(m => m.id === args.messageId)
      if (queued) agent.inbox.remove(args.messageId)
      const ownsTurn = boundary >= 0 && events.slice(boundary + 1).some(e => e.type === 'user/message' && (e.data.message || e.data).id === args.messageId && (e.data.message || e.data).source?.kind === 'user')
      if (agent.status === 'running' && !ownsTurn) return { confirmed: queued, foregroundOnly: true, reason: 'another_activity' }
      if (!queued && !events.some(e => e.type === 'user/message' && (e.data.message || e.data).id === args.messageId)) return { confirmed: false, reason: 'message_not_found' }
      agent.cancel({ kind: 'user' }, { keepInbox: true })
      let timer
      try {
        const settled = await Promise.race([agent.whenIdle().then(() => true), new Promise(resolve => { timer = setTimeout(() => resolve(false), 8500) })])
        return { confirmed: settled && agent.status !== 'running', foregroundOnly: true }
      } finally { clearTimeout(timer) }
    }
    throw new Error('Unsupported SDK control operation')
  }

  const server = http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json')
    try {
      const auth = Buffer.from(String(req.headers.authorization || ''))
      const wanted = Buffer.from('Bearer ' + secret)
      if (req.method !== 'POST' || req.url !== '/control' || auth.length !== wanted.length || !timingSafeEqual(auth, wanted)) { res.writeHead(403); res.end('{}'); return }
      const parts = []
      let size = 0
      for await (const part of req) { size += part.length; if (size > 65536) throw new Error('Request too large'); parts.push(part) }
      const payload = JSON.parse(Buffer.concat(parts).toString('utf8'))
      res.end(JSON.stringify({ result: await dispatch(payload.method, payload.args || {}) }))
    } catch (error) { res.writeHead(409); res.end(JSON.stringify({ error: String(error.message).slice(0, 200) })) }
  })
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve) })
  await fs.mkdir(config.root, { recursive: true })
  await fs.writeFile(endpoint, JSON.stringify({ port: server.address().port, pid: process.pid, contract: 1 }), { mode: 0o600 })
  ctx.effect(() => async () => {
    await new Promise(resolve => server.close(resolve))
    await fs.unlink(endpoint).catch(() => {})
  })
}
