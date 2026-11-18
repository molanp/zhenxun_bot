"""AI 聊天的配置项"""

from copy import deepcopy
from itertools import count
from typing import Any, Literal

from zhenxun.configs.utils import ConfigUIModel, RegisterConfig
from zhenxun.services.ai.run.dsh import DshSettings
from zhenxun.utils.pydantic_compat import model_dump

_ORDER = count(1)


def _options(**labels: str) -> list[dict[str, str]]:
    """配置项的可选值

    参数:
        labels: 值到 WebUI 显示名称的映射

    返回:
        list[dict[str, str]]: 可选值与显示名称
    """
    return [{"value": value, "label": label} for value, label in labels.items()]


def _when(
    path: str, value: Any, operator: Literal["eq", "neq"] = "eq"
) -> dict[str, Any]:
    """配置项在 WebUI 中的显示条件

    参数:
        path: 作为条件的配置键
        value: 比较的值
        operator: eq 等于时显示，neq 不等于时显示

    返回:
        dict[str, Any]: 显示条件
    """
    return {"path": path, "operator": operator, "value": value}


def _config(
    section: str,
    key: str,
    value: Any,
    type_: Any,
    label: str,
    description: str,
    **ui: Any,
) -> RegisterConfig:
    """AI 聊天配置项，WebUI 按分组和登记顺序展示

    参数:
        section: WebUI 中的分组
        key: 配置键
        value: 默认值
        type_: 值类型
        label: WebUI 中的名称
        description: 配置说明
        ui: 其余展示设置，如 options、unit、secret、advanced、visible_when

    返回:
        RegisterConfig: 配置项
    """
    return RegisterConfig(
        key=key,
        value=value,
        help=description,
        default_value=deepcopy(value),
        type=type_,
        ui=ConfigUIModel(label=label, section=section, order=next(_ORDER), **ui),
    )


CONFIGS = [
    _config(
        "请求",
        "TURN_TIMEOUT",
        300,
        int,
        "整轮时限",
        "取得执行权后，模型、工具与投递的总时限",
        unit="秒",
        min=1,
    ),
    _config(
        "唤醒",
        "WAKE_WORDS",
        [],
        list[str],
        "唤醒词",
        "消息以唤醒词开头时回复；@Bot、回复 Bot 的消息和私聊总会回复",
    ),
    _config(
        "唤醒",
        "SLASH_WAKE",
        True,
        bool,
        "/ 开头唤醒",
        "消息以 / 开头时回复，/ 不发给 AI；与已注册的指令同名时执行指令",
    ),
    _config(
        "回复",
        "CONTEXT_SIZE",
        20,
        int,
        "群聊背景消息数",
        "回复时一并交给 AI 的群聊近期消息条数，0 为不附带",
        unit="条",
    ),
    _config(
        "回复",
        "REPLY_QUOTE",
        False,
        bool,
        "引用触发消息",
        "群聊回复时引用触发回复的那条消息",
    ),
    _config(
        "回复",
        "SPLIT_REPLY",
        False,
        bool,
        "分段发送",
        "把回复按句拆成多条消息依次发送",
    ),
    _config(
        "回复",
        "SPLIT_INTERVAL",
        "1.5,3.5",
        str,
        "分段间隔",
        "每段发送前等待的秒数，写成“最小,最大”时随机取值",
        visible_when=_when("SPLIT_REPLY", True),
    ),
    _config(
        "回复",
        "SPLIT_MAX_CHARS",
        0,
        int,
        "分段字数上限",
        "回复超过该字数时整条发送、不再分段，0 为不限制",
        unit="字",
        visible_when=_when("SPLIT_REPLY", True),
    ),
    _config(
        "回复",
        "SPLIT_PATTERN",
        "",
        str,
        "分段规则",
        "正则表达式，在每个匹配之后切分，匹配内容保留在前一段；"
        "留空时按句末标点切分，换行始终切分",
        placeholder="[。！？!?]|哈哈+",
        visible_when=_when("SPLIT_REPLY", True),
    ),
    _config(
        "回复",
        "SPLIT_CLEANUP",
        "",
        str,
        "分段清理规则",
        "正则表达式，从每一段中删除匹配内容，如段尾的句号",
        placeholder="[。，]$",
        visible_when=_when("SPLIT_REPLY", True),
    ),
    _config(
        "回复",
        "LONG_REPLY_CHARS",
        1500,
        int,
        "长回复字数",
        "回复超过该字数时按长回复方式发送，0 为不区分长回复",
        unit="字",
    ),
    _config(
        "回复",
        "LONG_REPLY",
        "forward",
        str,
        "长回复方式",
        "长回复的发送方式；QQ 官方不支持合并转发，会发成图片",
        options=_options(text="直接发送", forward="合并转发", image="转为图片"),
    ),
    _config(
        "回复",
        "VOICE_REPLY",
        "off",
        str,
        "语音回复",
        "用语音合成默认模型把回复转成语音发送",
        options=_options(off="关闭", voice="用户发语音时", always="总是"),
    ),
    _config(
        "回复",
        "SEARCH_HINT",
        "",
        str,
        "搜索提示",
        "AI 联网搜索前先发送的一句话，如“我去搜一下，稍等～”，默认留空不发送；"
        "模型自带的联网搜索不会触发",
    ),
    _config(
        "回复",
        "REQUEST_TIMEOUT",
        60,
        int,
        "模型响应超时",
        "单次模型请求超过该时间未完成时换下一条线路，全部失败时提示用户；"
        "不超过服务商设置的超时，0 为使用服务商设置",
        unit="秒",
    ),
    _config(
        "主动插话",
        "PROACTIVE_MODE",
        "probability",
        str,
        "主动插话",
        "群里没人叫 Bot 时是否插话：按概率抽中后让 AI 决定说不说；监听模式每攒够"
        "一批新消息让监听模型判断该不该接话，更准但每次判断都消耗 token。"
        "不影响主动任务；可按群设置",
        options=_options(off="关闭", probability="概率", listen="监听"),
    ),
    _config(
        "主动插话",
        "ACTIVE_PROBABILITY",
        0.0,
        float,
        "插话概率",
        "概率模式下每条没叫 Bot 的群消息触发插话的概率（0~1），0 为不插话；"
        "可按群设置",
        visible_when=_when("PROACTIVE_MODE", "probability"),
    ),
    _config(
        "主动插话",
        "LISTEN_MESSAGES",
        5,
        int,
        "监听批量",
        "监听模式下群里攒够这么多条新消息才判断一次",
        unit="条",
        visible_when=_when("PROACTIVE_MODE", "listen"),
    ),
    _config(
        "主动插话",
        "LISTEN_INTERVAL",
        120,
        int,
        "监听间隔",
        "同一个群两次判断的最小间隔，0 为不限制",
        unit="秒",
        visible_when=_when("PROACTIVE_MODE", "listen"),
    ),
    _config(
        "主动插话",
        "LISTEN_MODEL",
        "",
        str,
        "监听模型",
        "判断是否接话的模型，写成“服务商/模型”，留空用对话默认模型；"
        "建议用便宜的小模型",
        visible_when=_when("PROACTIVE_MODE", "listen"),
    ),
    _config(
        "主动插话",
        "PROACTIVE_COOLDOWN",
        0,
        int,
        "插话冷却",
        "同一个群两次插话的最小间隔，0 为不限制；可按群设置",
        unit="秒",
        visible_when=_when("PROACTIVE_MODE", "off", "neq"),
    ),
    _config(
        "主动插话",
        "PROACTIVE_MAX_UNANSWERED",
        3,
        int,
        "没人理就停",
        "同一个群连续插话这么多次都没人 @ 或回复 Bot 时暂停插话，有人叫 Bot 后恢复；"
        "0 为不限制",
        unit="次",
        visible_when=_when("PROACTIVE_MODE", "off", "neq"),
    ),
    _config(
        "主动插话",
        "PROACTIVE_DAILY_COUNT",
        0,
        int,
        "每群每日插话次数",
        "每个群每天最多插话的次数，0 为不限制；可按群设置",
        unit="次",
        visible_when=_when("PROACTIVE_MODE", "off", "neq"),
    ),
    _config(
        "主动插话",
        "PROACTIVE_DAILY_TOKENS",
        0,
        int,
        "每日插话预算",
        "所有群的插话与监听判断每天共用的 token 上限，0 为不限制；"
        "每个群的预算可按群设置",
        unit="tokens",
        visible_when=_when("PROACTIVE_MODE", "off", "neq"),
    ),
    _config(
        "主动插话",
        "PROACTIVE_QUIET_HOURS",
        "",
        str,
        "免打扰时段",
        "这段时间不插话，写成“开始-结束”的小时，如 23-7；留空为不限制",
        placeholder="23-7",
        visible_when=_when("PROACTIVE_MODE", "off", "neq"),
    ),
    _config(
        "偷表情",
        "STICKER_STEAL",
        "off",
        str,
        "偷表情",
        "自动收集群里的表情包（商城表情和协议端标为表情的图片）：概率模式抽样候选，"
        "监听模式检查每个候选；两种模式均由视觉模型确认表情属性与收藏价值后入库，"
        "同时写好备注，每次判断消耗 token。可按群设置",
        options=_options(off="关闭", probability="概率", listen="监听"),
    ),
    _config(
        "偷表情",
        "STEAL_PROBABILITY",
        0.1,
        float,
        "偷表情概率",
        "概率模式下每个表情包被选为 AI 判断候选的概率（0~1）",
        visible_when=_when("STICKER_STEAL", "probability"),
    ),
    _config(
        "偷表情",
        "STEAL_MODEL",
        "",
        str,
        "判断模型",
        "判断表情属性与收藏价值的视觉模型，写成“服务商/模型”，留空用视觉默认模型",
        visible_when=_when("STICKER_STEAL", "off", "neq"),
    ),
    _config(
        "偷表情",
        "STEAL_DAILY_JUDGES",
        50,
        int,
        "每日判断次数",
        "所有群每天最多让模型判断的次数，0 为不限制",
        unit="次",
        visible_when=_when("STICKER_STEAL", "off", "neq"),
    ),
    _config(
        "偷表情",
        "STEAL_COOLDOWN",
        300,
        int,
        "偷表情冷却",
        "同一个群两次收集的最小间隔，0 为不限制",
        unit="秒",
        visible_when=_when("STICKER_STEAL", "off", "neq"),
    ),
    _config(
        "偷表情",
        "STEAL_DAILY_COUNT",
        10,
        int,
        "每群每日收集数",
        "每个群每天最多收集的表情数，0 为不限制；表情库上限见“发送与表情”",
        unit="张",
        visible_when=_when("STICKER_STEAL", "off", "neq"),
    ),
    _config(
        "发送与表情",
        "SEND_TOOLS",
        True,
        bool,
        "发送类工具",
        "允许 AI 在当前会话里另外发消息、戳一戳、发表情",
    ),
    _config(
        "发送与表情",
        "SEND_PER_MINUTE",
        8,
        int,
        "每分钟发送上限",
        "每个会话里 AI 每分钟最多使用发送类工具的次数，0 为不限",
        unit="次",
        visible_when=_when("SEND_TOOLS", True),
    ),
    _config(
        "发送与表情",
        "SEND_PER_HOUR",
        60,
        int,
        "每小时发送上限",
        "每个会话里 AI 每小时最多使用发送类工具的次数，0 为不限",
        unit="次",
        visible_when=_when("SEND_TOOLS", True),
    ),
    _config(
        "发送与表情",
        "STICKER_FREQUENCY",
        "balanced",
        str,
        "表情发送频率",
        "通过提示词控制 AI 使用表情的倾向，不改变发送安全配额",
        options=_options(high="高", balanced="均衡", low="低"),
    ),
    _config(
        "发送与表情",
        "STICKER_COLLECT",
        True,
        bool,
        "收藏表情",
        "允许 AI 收藏群里的表情包，一次可收藏同一条消息里的多张",
    ),
    _config(
        "发送与表情",
        "STICKER_PER_MINUTE",
        5,
        int,
        "每分钟收藏上限",
        "每个会话里 AI 每分钟最多收藏的表情数，按张计算，0 为不限",
        unit="张",
        visible_when=_when("STICKER_COLLECT", True),
    ),
    _config(
        "发送与表情",
        "STICKER_PER_HOUR",
        20,
        int,
        "每小时收藏上限",
        "每个会话里 AI 每小时最多收藏的表情数，按张计算，0 为不限",
        unit="张",
        visible_when=_when("STICKER_COLLECT", True),
    ),
    _config(
        "发送与表情",
        "STICKER_LIMIT",
        200,
        int,
        "表情库上限",
        "表情库最多保存的表情数，0 为不限",
        unit="张",
    ),
    _config(
        "发送与表情",
        "STICKER_AUTO_DESCRIBE",
        False,
        bool,
        "自动补写表情备注",
        "在后台给没有备注的表情补写备注、标签和适用场合，"
        "使用视觉默认模型（未配置时用对话模型）",
    ),
    _config(
        "发送与表情",
        "STICKER_QQ_SYNC",
        False,
        bool,
        "同步到 QQ 收藏",
        "AI 收藏表情时同时加入 Bot 账号的 QQ 收藏表情"
        "（需要协议端支持 add_custom_face，如 SnowLuma）",
    ),
    _config(
        "长期记忆",
        "MEMORY_MODE",
        "off",
        str,
        "长期记忆",
        "把群聊和私聊整理成记忆片段，之后聊到相关内容时召回。自动召回：每次对话前"
        "按消息召回并附给 AI；按需召回：由 AI 调用回忆工具。两种方式 AI 都能记下和"
        "忘掉记忆；群记忆整群共享，私聊记忆按用户区分",
        options=_options(off="关闭", inject="自动召回", tool="按需召回"),
    ),
    _config(
        "长期记忆",
        "MEMORY_TURNS",
        10,
        int,
        "整理间隔",
        "会话里每攒够这么多条新消息，在后台把这段对话整理成一条记忆",
        unit="条",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
    ),
    _config(
        "长期记忆",
        "MEMORY_TOP_K",
        5,
        int,
        "召回条数",
        "每次最多召回的记忆条数",
        unit="条",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
    ),
    _config(
        "长期记忆",
        "MEMORY_DECAY",
        0.01,
        float,
        "每日衰减",
        "自动整理的记忆每天降低的重要度，被召回时会回升，降到 0.05 以下时归档；"
        "0 为不衰减",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
    ),
    _config(
        "长期记忆",
        "MEMORY_MODEL",
        "",
        str,
        "整理模型",
        "整理记忆的模型，写成“服务商/模型”，留空用对话默认模型",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
        advanced=True,
    ),
    _config(
        "长期记忆",
        "MEMORY_EMBEDDING",
        True,
        bool,
        "向量召回",
        "用 embedding 模型计算记忆向量，按语义加关键词召回；关闭后只用关键词召回，"
        "不再调用 embedding 模型",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
    ),
    _config(
        "长期记忆",
        "MEMORY_EMBEDDING_MODEL",
        "",
        str,
        "记忆向量模型",
        "计算记忆向量的 embedding 模型，留空用默认 embedding 模型，都没有时只用"
        "关键词召回；更换后在“长期记忆”页重算向量",
        visible_when=_when("MEMORY_MODE", "off", "neq"),
        advanced=True,
    ),
    _config(
        "知识库与角色识别",
        "KNOWLEDGE_MODE",
        "inject",
        str,
        "知识库检索方式",
        "自动检索：每次对话前按消息检索并附给 AI；按需检索：由 AI 调用检索工具",
        options=_options(inject="自动检索", tool="按需检索"),
    ),
    _config(
        "知识库与角色识别",
        "CHARACTER_RECOGNITION",
        "off",
        str,
        "角色识别",
        "开启后 AI 可按需识别图片中的二次元角色；本地模型在“角色识别”页安装",
        options=_options(
            off="关闭",
            cloud="仅云端",
            lite="轻量本地模型 + 云端",
            full="完整本地模型 + 云端",
        ),
    ),
    _config(
        "知识库与角色识别",
        "CHARACTER_MIN_SCORE",
        0.5,
        float,
        "最低识别分",
        "低于该分数的候选不采用；分数为各来源分数按权重相加",
        visible_when=_when("CHARACTER_RECOGNITION", "off", "neq"),
    ),
    _config(
        "知识库与角色识别",
        "CHARACTER_SAUCENAO_KEY",
        "",
        str,
        "SauceNAO API Key",
        "填写后增加 SauceNAO 以图搜图来源",
        secret=True,
        visible_when=_when("CHARACTER_RECOGNITION", "off", "neq"),
    ),
    _config(
        "主动任务",
        "TASK_LIMIT",
        20,
        int,
        "任务总数上限",
        "对话中创建的定时、延时任务的总数上限",
        unit="个",
    ),
    _config(
        "主动任务",
        "TASK_USER_LIMIT",
        5,
        int,
        "每人任务上限",
        "每位用户最多创建的任务数，0 为不限制；超级用户不受限制",
        unit="个",
    ),
    _config(
        "Agent 与执行器",
        "AGENT_ENABLED",
        False,
        bool,
        "超级用户 Agent",
        "为超级用户提供 HTTP 请求、沙箱、MCP 等工具，调用记录在 Agent 页；"
        "普通用户不会获得这些工具",
    ),
    _config(
        "Agent 与执行器",
        "AGENT_RUNNER",
        "local",
        str,
        "回复执行器",
        "DeepSeek Harness 可选择已登录的桌面版或独立 SDK；普通聊天无宿主工具，"
        "不桥接长期记忆、知识库及本地 Agent 工具；"
        "账号场景设置优先于按群和全局设置；"
        "可用 /ai runner dsh 切换当前场景，/ai status 查看实际执行器",
        options=_options(local="本地 Agent", dsh="DeepSeek Harness"),
    ),
    _config(
        "额度与安全",
        "DAILY_TOKEN_LIMIT",
        0,
        int,
        "每人每日 token",
        "每位用户每天可用的 token 上限，0 为不限制",
        unit="tokens",
    ),
    _config(
        "额度与安全",
        "COST_GOLD",
        0,
        int,
        "每次消耗金币",
        "每次对话扣除的金币，0 为免费",
        unit="金币",
    ),
    _config(
        "额度与安全",
        "USER_RATE_LIMIT",
        10,
        int,
        "每人每分钟次数",
        "每位用户每分钟最多唤醒的次数，超出时只提示一次；0 为不限制，"
        "超级用户不受限制",
        unit="次",
    ),
    _config(
        "额度与安全",
        "SESSION_RATE_LIMIT",
        30,
        int,
        "每个会话每分钟次数",
        "每个群或私聊每分钟最多回复的次数，超出的排队等待，"
        "排队超过 10 条时丢弃；"
        "0 为不限制",
        unit="次",
    ),
    _config(
        "额度与安全",
        "SAFETY_WORDS",
        [],
        list[str],
        "安全词",
        "用户消息命中时不调用 AI，AI 回复命中时按“安全词处理”处理；写成 /…/ 为正则",
    ),
    _config(
        "额度与安全",
        "SAFETY_MODE",
        "block",
        str,
        "安全词处理",
        "AI 回复命中安全词时的处理方式",
        options=_options(block="整条拦截", mask="命中部分替换为 *"),
    ),
    _config(
        "知识库与角色识别",
        "KNOWLEDGE_BASES",
        [],
        list[int],
        "全局知识库",
        "没有单独设置的群和私聊使用的知识库 id，在“知识库”页管理",
        advanced=True,
    ),
    _config(
        "知识库与角色识别",
        "CHARACTER_SOURCES",
        {},
        dict,
        "角色识别来源",
        "各识别来源的开关与权重，在“角色识别”页调整",
        advanced=True,
    ),
    _config(
        "知识库与角色识别",
        "HF_ENDPOINT",
        "https://hf-mirror.com",
        str,
        "Hugging Face 地址",
        "下载角色识别模型与角色库的地址，可换成镜像",
        advanced=True,
    ),
    _config(
        "Agent 与执行器",
        "DISABLED_SKILLS",
        [],
        list[str],
        "停用的技能",
        "停用的技能名，在“技能”页管理；技能文件位于 data/ai/skills/<名称>/SKILL.md",
        advanced=True,
    ),
    _config(
        "Agent 与执行器",
        "DSH",
        model_dump(DshSettings()),
        dict,
        "DeepSeek Harness 设置",
        "transport：desktop 或 sdk，不自动互相回退。"
        "desktop_url：桌面回环 HTTP 地址，留空自动识别；home：DSH home，"
        "留空使用当前用户 .dsh；timeout：每轮超时，等待人工确认的时间不计入；"
        "permission：Agent 权限，sandbox 在沙箱内执行，越权操作需超管批准，"
        "可回复“全部允许”批准本轮后续请求；full 不使用沙箱且不再询问；"
        "allow_tools：允许符合"
        "当前授权的超管通过 QQ 操作 DSH 原生 Agent，不另加工具白名单。"
        "桌面无需 API Key；sdk_bin 指向已安装 SDK，sdk_home 为独立宿主目录，"
        "sdk_provider 引用已配置的官方 DeepSeek 提供商，"
        "sdk_model、sdk_thinking 为新会话默认选择。"
        "SDK 原生 Agent 在 Bot 退出后继续运行；只停止转发，不补发旧消息。"
        "两种后端分别继承会话，不迁移或共用桌面运行目录。"
        "超管 Agent 跨群和私聊共享历史，完整回复发到本轮触发位置，"
        "群内回复可能包含此前私聊或工具读取的信息。",
        advanced=True,
        visible_when=_when("AGENT_RUNNER", "dsh"),
    ),
]
"""AI 聊天的配置项，WebUI 按分组和登记顺序展示"""
