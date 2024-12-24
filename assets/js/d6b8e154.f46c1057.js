"use strict";(self.webpackChunkzhenxun_docs=self.webpackChunkzhenxun_docs||[]).push([[9726],{6103:(n,e,r)=>{r.r(e),r.d(e,{assets:()=>a,contentTitle:()=>t,default:()=>c,frontMatter:()=>i,metadata:()=>d,toc:()=>l});var s=r(4848),o=r(8453);const i={title:"\u5546\u5e97\u914d\u7f6e",subSidebar:!1},t=void 0,d={id:"development/shop",title:"\u5546\u5e97\u914d\u7f6e",description:"\u6982\u89c8",source:"@site/docs/development/shop.md",sourceDirName:"development",slug:"/development/shop",permalink:"/zhenxun_bot/development/shop",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/development/shop.md",tags:[],version:"current",frontMatter:{title:"\u5546\u5e97\u914d\u7f6e",subSidebar:!1},sidebar:"devSidebar",previous:{title:"\u914d\u7f6e\u7ba1\u7406",permalink:"/zhenxun_bot/development/config"},next:{title:"\u6d88\u606f\u53d1\u9001",permalink:"/zhenxun_bot/development/message"}},a={},l=[{value:"\u6982\u89c8",id:"\u6982\u89c8",level:2},{value:"\u65b9\u6cd5\u4ecb\u7ecd",id:"\u65b9\u6cd5\u4ecb\u7ecd",level:2},{value:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570",id:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570",level:3},{value:"\u9053\u5177\u4f7f\u7528\u524d\u51fd\u6570",id:"\u9053\u5177\u4f7f\u7528\u524d\u51fd\u6570",level:3},{value:"\u9053\u5177\u4f7f\u7528\u540e\u51fd\u6570",id:"\u9053\u5177\u4f7f\u7528\u540e\u51fd\u6570",level:3},{value:"\u591a\u4e2a\u9053\u5177\u4f7f\u7528\u540c\u4e00\u4e2a\u6ce8\u518c\u51fd\u6570\uff08\u4ee5\u7b7e\u5230\u4e3a\u6817\u5b50\uff09",id:"\u591a\u4e2a\u9053\u5177\u4f7f\u7528\u540c\u4e00\u4e2a\u6ce8\u518c\u51fd\u6570\u4ee5\u7b7e\u5230\u4e3a\u6817\u5b50",level:2},{value:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570",id:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570-1",level:3},{value:"\u9053\u5177\u4f7f\u7528\u524d/\u540e\u51fd\u6570",id:"\u9053\u5177\u4f7f\u7528\u524d\u540e\u51fd\u6570",level:3}];function p(n){const e={br:"br",code:"code",h2:"h2",h3:"h3",p:"p",pre:"pre",...(0,o.R)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.h2,{id:"\u6982\u89c8",children:"\u6982\u89c8"}),"\n",(0,s.jsx)(e.p,{children:"\u6dfb\u52a0\u5546\u5e97\u7684\u5546\u54c1\u4fe1\u606f\u4e0e\u4f7f\u7528\u51fd\u6570"}),"\n",(0,s.jsx)(e.h2,{id:"\u65b9\u6cd5\u4ecb\u7ecd",children:"\u65b9\u6cd5\u4ecb\u7ecd"}),"\n",(0,s.jsx)(e.h3,{id:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570",children:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'\nfrom nonebot_plugin_alconna import UniMessage, UniMsg\nfrom nonebot_plugin_uninfo import Uninfo\n\n@shop_register(\n    name="\u6d4b\u8bd5\u9053\u5177A",\n    price=99,\n    des="\u968f\u4fbf\u4fa7\u800c\u51fa",\n    load_status=False,\n    icon="sword.png",\n    **{"prob": 100}             # \u81ea\u5b9a\u4e49\u53c2\u6570\u4f20\u9012\n)\nasync def _(\n    user_id: str,               # \u7528\u6237id\n    group_id: str | None,       # \u7fa4\u7ec4id\n    bot: Bot,                   # Bot\u5b9e\u4f8b\n    event: Event,               # event\u5b9e\u4f8b\n    num: int,                   # \u9053\u5177\u4f7f\u7528\u4e2a\u6570\n    send_success_msg: bool,     # \u662f\u5426\u53d1\u9001\u4f7f\u7528\u6210\u529f\u6d88\u606f\n    max_num_limit: int,         # \u6700\u5927\u4f7f\u7528\u4e2a\u6570\u9650\u5236\n    session: Uninfo,            # Uninfo\u5b9e\u4f8b\n    message: UniMsg             # UniMsg\u5b9e\u4f8b\n    prob: int                   # \u81ea\u5b9a\u4e49\u53c2\u6570\n):\n    print(user_id, group_id, "\u4f7f\u7528\u6d4b\u8bd5\u9053\u5177")\n'})}),"\n",(0,s.jsxs)(e.p,{children:["\u4f7f\u7528\u65b9\u6cd5\u4e2d\u7684\u53c2\u6570\u53ef\u4ee5\u6839\u636e\u81ea\u8eab\u9700\u6c42\u6765",(0,s.jsx)(e.br,{}),"\n","\u4f8b\u5982\u5f53\u4ec5\u9700\u8981\u7528\u6237id\u548c\u7fa4\u7ec4id\u65f6\uff0c\u53ef\u4ee5\u4f7f\u7528\u4ee5\u4e0b"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'\nfrom nonebot_plugin_alconna import UniMessage, UniMsg\nfrom nonebot_plugin_uninfo import Uninfo\n\n@shop_register(\n    name="\u6d4b\u8bd5\u9053\u5177A",\n    price=99,\n    des="\u968f\u4fbf\u4fa7\u800c\u51fa",\n    load_status=True,           # \u4e3aFalse\u65f6\u4e0d\u52a0\u8f7d\u8be5\u51fd\u6570\n    icon="sword.png",\n    **{"prob": 100}             # \u81ea\u5b9a\u4e49\u53c2\u6570\u4f20\u9012\n)\nasync def _(user_id: str, group_id: str | None):\n    print(user_id, group_id, "\u4f7f\u7528\u6d4b\u8bd5\u9053\u5177")\n'})}),"\n",(0,s.jsx)(e.h3,{id:"\u9053\u5177\u4f7f\u7528\u524d\u51fd\u6570",children:"\u9053\u5177\u4f7f\u7528\u524d\u51fd\u6570"}),"\n",(0,s.jsxs)(e.p,{children:["\u51fd\u6570\u53c2\u6570\u89c4\u5219\u4e0e\u6ce8\u518c\u51fd\u6570\u76f8\u540c",(0,s.jsx)(e.br,{}),"\n","\u9053\u5177\u540d\u79f0\u4f5c\u4e3a\u952e\u503c\uff0c\u76f8\u540c\u9053\u5177\u53ef\u4ee5\u62e5\u6709\u591a\u4e2a\u4f7f\u7528\u524d\u51fd\u6570",(0,s.jsx)(e.br,{}),"\n","\u53ef\u4ee5\u629b\u51fa",(0,s.jsx)(e.code,{children:"NotMeetUseConditionsException"}),"\u5f02\u5e38\uff0c\u963b\u65ad\u4f7f\u7528\uff0c\u5e76\u8fd4\u56de\u4fe1\u606f"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'from zhenxun.utils.decorator.shop import NotMeetUseConditionsException\n\nfrom nonebot_plugin_alconna import UniMessage, UniMsg\nfrom nonebot_plugin_uninfo import Uninfo\n\n@shop_register.before_handle(name="\u6d4b\u8bd5\u9053\u5177A")\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e00\u4e2a\u4f7f\u7528\u524d\u51fd\u6570\uff08before handle\uff09")\n\n\n@shop_register.before_handle(name="\u6d4b\u8bd5\u9053\u5177A", load_status=False)   # load_status\u4e3aFalse\uff0c\u4e0d\u52a0\u8f7d\u8be5\u51fd\u6570\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e8c\u4e2a\u4f7f\u7528\u524d\u51fd\u6570\uff08before handle\uff09")\n    raise NotMeetUseConditionsException("\u592a\u7b28\u4e86\uff01")  # \u629b\u51fa\u5f02\u5e38\uff0c\u963b\u65ad\u4f7f\u7528\uff0c\u5e76\u8fd4\u56de\u4fe1\u606f\n'})}),"\n",(0,s.jsx)(e.h3,{id:"\u9053\u5177\u4f7f\u7528\u540e\u51fd\u6570",children:"\u9053\u5177\u4f7f\u7528\u540e\u51fd\u6570"}),"\n",(0,s.jsxs)(e.p,{children:["\u4e0e\u4f7f\u7528\u524d\u51fd\u6570\u76f8\u540c\uff0c\u4f46\u65e0\u6cd5\u629b\u51fa",(0,s.jsx)(e.code,{children:"NotMeetUseConditionsException"})]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'@shop_register.after_handle(name="\u6d4b\u8bd5\u9053\u5177A")\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e00\u4e2a\u4f7f\u7528\u540e\u51fd\u6570\uff08after handle\uff09")\n\n@shop_register.after_handle(name="\u6d4b\u8bd5\u9053\u5177A")\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e8c\u4e2a\u4f7f\u7528\u540e\u51fd\u6570\uff08after handle\uff09")\n'})}),"\n",(0,s.jsx)(e.h2,{id:"\u591a\u4e2a\u9053\u5177\u4f7f\u7528\u540c\u4e00\u4e2a\u6ce8\u518c\u51fd\u6570\u4ee5\u7b7e\u5230\u4e3a\u6817\u5b50",children:"\u591a\u4e2a\u9053\u5177\u4f7f\u7528\u540c\u4e00\u4e2a\u6ce8\u518c\u51fd\u6570\uff08\u4ee5\u7b7e\u5230\u4e3a\u6817\u5b50\uff09"}),"\n",(0,s.jsx)(e.p,{children:"\u53ef\u4ee5\u4f7f\u7528\u5143\u7956\u53c2\u6570\u6765\u4f20\u9012\u591a\u4e2a\u9053\u5177\u540d\u79f0\u548c\u914d\u7f6e"}),"\n",(0,s.jsx)(e.h3,{id:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570-1",children:"\u4f7f\u7528\u65b9\u6cd5\u6ce8\u518c\u51fd\u6570"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'from nonebot_plugin_alconna import UniMessage, UniMsg\nfrom nonebot_plugin_uninfo import Uninfo\n\n@shop_register(\n    name=("\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2160", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2161", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2162"),\n    price=(30, 150, 250),\n    des=(\n        "\u4e0b\u6b21\u7b7e\u5230\u53cc\u500d\u597d\u611f\u5ea6\u6982\u7387 + 10%\uff08\u8c01\u624d\u662f\u771f\u547d\u5929\u5b50\uff1f\uff09\uff08\u540c\u7c7b\u5546\u54c1\u5c06\u8986\u76d6\uff09",\n        "\u4e0b\u6b21\u7b7e\u5230\u53cc\u500d\u597d\u611f\u5ea6\u6982\u7387 + 20%\uff08\u5e73\u5e73\u5eb8\u5eb8\uff09\uff08\u540c\u7c7b\u5546\u54c1\u5c06\u8986\u76d6\uff09",\n        "\u4e0b\u6b21\u7b7e\u5230\u53cc\u500d\u597d\u611f\u5ea6\u6982\u7387 + 30%\uff08\u91d1\u5e01\u624d\u662f\u771f\u547d\u5929\u5b50\uff01\uff09\uff08\u540c\u7c7b\u5546\u54c1\u5c06\u8986\u76d6\uff09",\n    ),\n    load_status=True,\n    icon=(\n        "favorability_card_1.png",\n        "favorability_card_2.png",\n        "favorability_card_3.png",\n    ),\n    **{\n        "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2160_prob": 0.1,\n        "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2161_prob": 0.2,\n        "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2162_prob": 0.3,\n    },  # \u81ea\u5b9a\u4e49\u53c2\u6570\uff0c\u591a\u4e2a\u9053\u5177\u65f6\u9700\u8981\u683c\u5f0f\u4e3a  \u9053\u5177\u540d\u79f0 + \u53c2\u6570\u540d\u79f0\n)\nasync def _(session: Uninfo, user_id: int, prob: float):\n    """\n    prob\u4f1a\u6839\u636e\u4f7f\u7528\u7684\u9053\u5177\u4e0d\u540c\u800c\u4e0d\u540c\n    \u5f53\u4f7f\u7528\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2160\u65f6prob\u4e3a0.1\n    \u5f53\u4f7f\u7528\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2161\u65f6prob\u4e3a0.2\uff0c\u4ee5\u6b64\u7c7b\u63a8\n    """\n    pass\n'})}),"\n",(0,s.jsx)(e.h3,{id:"\u9053\u5177\u4f7f\u7528\u524d\u540e\u51fd\u6570",children:"\u9053\u5177\u4f7f\u7528\u524d/\u540e\u51fd\u6570"}),"\n",(0,s.jsx)(e.p,{children:"\u4f7f\u7528\u65b9\u6cd5\u4e0e\u5355\u4e2a\u4f7f\u7528\u76f8\u540c\uff0c\u53ea\u662f\u53c2\u6570\u4f7f\u7528\u5143\u7956\u4f20\u9012"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:'from zhenxun.utils.decorator.shop import NotMeetUseConditionsException\n\n@shop_register.before_handle(name=("\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2160", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2161", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2162"))\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e00\u4e2a\u4f7f\u7528\u524d\u51fd\u6570\uff08before handle\uff09")\n\n@shop_register.after_handle(name=("\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2160", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2161", "\u597d\u611f\u5ea6\u53cc\u500d\u52a0\u6301\u5361\u2162"))\nasync def _(user_id: str, group_id: str):\n    print(user_id, group_id, "\u7b2c\u4e00\u4e2a\u4f7f\u7528\u540e\u51fd\u6570\uff08after handle\uff09")\n\n'})})]})}function c(n={}){const{wrapper:e}={...(0,o.R)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(p,{...n})}):p(n)}},8453:(n,e,r)=>{r.d(e,{R:()=>t,x:()=>d});var s=r(6540);const o={},i=s.createContext(o);function t(n){const e=s.useContext(i);return s.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function d(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(o):n.components||o:t(n.components),s.createElement(i.Provider,{value:e},n.children)}}}]);