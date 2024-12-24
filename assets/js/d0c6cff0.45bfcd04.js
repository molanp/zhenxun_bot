"use strict";(self.webpackChunkzhenxun_docs=self.webpackChunkzhenxun_docs||[]).push([[2833],{8492:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>N,contentTitle:()=>S,default:()=>E,frontMatter:()=>z,metadata:()=>w,toc:()=>I});var r=n(4848),l=n(8453),s=n(6540),i=n(4164),a=n(6641),o=n(6347),c=n(205),d=n(8874);var u=n(2993);function h(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:t}=e;return!!t&&"object"==typeof t&&"value"in t}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function x(e){const{values:t,children:n}=e;return(0,s.useMemo)((()=>{const e=t??function(e){return h(e).map((e=>{let{props:{value:t,label:n,attributes:r,default:l}}=e;return{value:t,label:n,attributes:r,default:l}}))}(n);return function(e){const t=(void 0===(r=(e,t)=>e.value===t.value)&&(r=(e,t)=>e===t),(n=e).filter(((e,t)=>n.findIndex((t=>r(t,e)))!==t)));var n,r;if(t.length>0)throw new Error(`Docusaurus error: Duplicate values "${t.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[t,n])}function p(e){let{value:t,tabValues:n}=e;return n.some((e=>e.value===t))}function b(e){let{queryString:t=!1,groupId:n}=e;const r=(0,o.W6)(),l=function(e){let{queryString:t=!1,groupId:n}=e;if("string"==typeof t)return t;if(!1===t)return null;if(!0===t&&!n)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return n??null}({queryString:t,groupId:n});return[(0,d.aZ)(l),(0,s.useCallback)((e=>{if(!l)return;const t=new URLSearchParams(r.location.search);t.set(l,e),r.replace({...r.location,search:t.toString()})}),[l,r])]}function g(e){const{defaultValue:t,queryString:n=!1,groupId:r}=e,l=x(e),[i,a]=(0,s.useState)((()=>function(e){let{defaultValue:t,tabValues:n}=e;if(0===n.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(t){if(!p({value:t,tabValues:n}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${t}" but none of its children has the corresponding value. Available values are: ${n.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return t}const r=n.find((e=>e.default))??n[0];if(!r)throw new Error("Unexpected error: 0 tabValues");return r.value}({defaultValue:t,tabValues:l}))),[o,d]=b({queryString:n,groupId:r}),[h,g]=function(e){let{groupId:t}=e;const n=function(e){return e?`docusaurus.tab.${e}`:null}(t),[r,l]=(0,u.Dv)(n);return[r,(0,s.useCallback)((e=>{n&&l.set(e)}),[n,l])]}({groupId:r}),j=(()=>{const e=o??h;return p({value:e,tabValues:l})?e:null})();(0,c.A)((()=>{j&&a(j)}),[j]);return{selectedValue:i,selectValue:(0,s.useCallback)((e=>{if(!p({value:e,tabValues:l}))throw new Error(`Can't select invalid tab value=${e}`);a(e),d(e),g(e)}),[d,g,l]),tabValues:l}}var j=n(2303);const m={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};function f(e){let{className:t,block:n,selectedValue:l,selectValue:s,tabValues:o}=e;const c=[],{blockElementScrollPositionUntilNextRender:d}=(0,a.a_)(),u=e=>{const t=e.currentTarget,n=c.indexOf(t),r=o[n].value;r!==l&&(d(t),s(r))},h=e=>{let t=null;switch(e.key){case"Enter":u(e);break;case"ArrowRight":{const n=c.indexOf(e.currentTarget)+1;t=c[n]??c[0];break}case"ArrowLeft":{const n=c.indexOf(e.currentTarget)-1;t=c[n]??c[c.length-1];break}}t?.focus()};return(0,r.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,i.A)("tabs",{"tabs--block":n},t),children:o.map((e=>{let{value:t,label:n,attributes:s}=e;return(0,r.jsx)("li",{role:"tab",tabIndex:l===t?0:-1,"aria-selected":l===t,ref:e=>c.push(e),onKeyDown:h,onClick:u,...s,className:(0,i.A)("tabs__item",m.tabItem,s?.className,{"tabs__item--active":l===t}),children:n??t},t)}))})}function v(e){let{lazy:t,children:n,selectedValue:l}=e;const a=(Array.isArray(n)?n:[n]).filter(Boolean);if(t){const e=a.find((e=>e.props.value===l));return e?(0,s.cloneElement)(e,{className:(0,i.A)("margin-top--md",e.props.className)}):null}return(0,r.jsx)("div",{className:"margin-top--md",children:a.map(((e,t)=>(0,s.cloneElement)(e,{key:t,hidden:e.props.value!==l})))})}function y(e){const t=g(e);return(0,r.jsxs)("div",{className:(0,i.A)("tabs-container",m.tabList),children:[(0,r.jsx)(f,{...t,...e}),(0,r.jsx)(v,{...t,...e})]})}function q(e){const t=(0,j.A)();return(0,r.jsx)(y,{...e,children:h(e.children)},String(t))}const A={tabItem:"tabItem_Ymn6"};function _(e){let{children:t,hidden:n,className:l}=e;return(0,r.jsx)("div",{role:"tabpanel",className:(0,i.A)(A.tabItem,l),hidden:n,children:t})}const z={title:"\u5b89\u88c5\u53ef\u7231\u7684\u5c0f\u771f\u5bfb",subSidebar:!1,id:"install-zhenxun"},S=void 0,w={id:"install/install-zhenxun",title:"\u5b89\u88c5\u53ef\u7231\u7684\u5c0f\u771f\u5bfb",description:"\u771f\u5bfbBot\u9700\u8981python\u7248\u672c\u4e3a 3.10\u62163.11",source:"@site/docs/install/zhenxun.md",sourceDirName:"install",slug:"/install/install-zhenxun",permalink:"/zhenxun_bot/install/install-zhenxun",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/install/zhenxun.md",tags:[],version:"current",frontMatter:{title:"\u5b89\u88c5\u53ef\u7231\u7684\u5c0f\u771f\u5bfb",subSidebar:!1,id:"install-zhenxun"},sidebar:"installSidebar",previous:{title:"\u5b89\u88c5QQ",permalink:"/zhenxun_bot/install/install-qq"},next:{title:"\u9ed8\u8ba4\u5b89\u88c5",permalink:"/zhenxun_bot/install/webui/install-webui-default"}},N={},I=[{value:"\u4e0b\u8f7d",id:"\u4e0b\u8f7d",level:2},{value:"\u5b89\u88c5\u4f9d\u8d56\u5305",id:"\u5b89\u88c5\u4f9d\u8d56\u5305",level:2},{value:"\u57fa\u7840\u914d\u7f6e",id:"\u57fa\u7840\u914d\u7f6e",level:2},{value:"\u8bbe\u7f6e\u8d85\u7ea7\u7528\u6237\uff0c\u6253\u5f00 <strong>.env.dev</strong> \u6587\u4ef6\uff0c\u5728<code>SUPERUSERS</code>\u548c<code>qq</code>\u4e2d\u6dfb\u52a0\u81ea\u5df1\u7684QQ",id:"\u8bbe\u7f6e\u8d85\u7ea7\u7528\u6237\u6253\u5f00-envdev-\u6587\u4ef6\u5728superusers\u548cqq\u4e2d\u6dfb\u52a0\u81ea\u5df1\u7684qq",level:4},{value:"\u6570\u636e\u5e93\u914d\u7f6e",id:"\u6570\u636e\u5e93\u914d\u7f6e",level:4},{value:"\u57fa\u7840\u63d2\u4ef6\u914d\u7f6e",id:"\u57fa\u7840\u63d2\u4ef6\u914d\u7f6e",level:4},{value:"\u542f\u52a8",id:"\u542f\u52a8",level:2},{value:"\u5f53\u4f60\u7684\u63a7\u5236\u53f0\u51fa\u73b0\u4ee5\u4e0b\u65e5\u5fd7\uff0c\u8bf4\u660e\u4f60\u5df2\u7ecf\u6210\u529f\u4e86\ud83c\udf89\ud83c\udf89",id:"\u5f53\u4f60\u7684\u63a7\u5236\u53f0\u51fa\u73b0\u4ee5\u4e0b\u65e5\u5fd7\u8bf4\u660e\u4f60\u5df2\u7ecf\u6210\u529f\u4e86",level:4}];function k(e){const t={a:"a",admonition:"admonition",code:"code",h2:"h2",h4:"h4",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,l.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsxs)(t.admonition,{title:"\u7248\u672c\u8b66\u544a",type:"warning",children:[(0,r.jsxs)(t.p,{children:["\u771f\u5bfbBot\u9700\u8981python\u7248\u672c\u4e3a ",(0,r.jsx)(t.strong,{children:"3.10\u62163.11"})]}),(0,r.jsx)(t.admonition,{title:"Tip",type:"info",children:(0,r.jsxs)(t.p,{children:[(0,r.jsx)(t.strong,{children:"Python3.9"})," \u540c\u6837\u4e5f\u53ef\u4ee5\u4f7f\u7528\uff0c\u4f46\u662f\u9700\u8981\u5c06pyproject.toml\u4e2d\u7684\u7248\u672c\u6539\u4e3a3.9"]})})]}),"\n",(0,r.jsx)(t.h2,{id:"\u4e0b\u8f7d",children:"\u4e0b\u8f7d"}),"\n",(0,r.jsxs)(t.p,{children:["\u4ece ",(0,r.jsx)(t.a,{href:"https://github.com/HibiKier/zhenxun_bot",children:"HibiKier / zhenxun_bot"})," clone\u4ee3\u7801 \u6216 \u76f4\u63a5\u4e0b\u8f7d ",(0,r.jsx)(t.a,{href:"https://github.com/HibiKier/zhenxun_bot/archive/refs/heads/main.zip",children:"\u538b\u7f29\u5305"})," \u89e3\u538b"]}),"\n",(0,r.jsx)(t.h2,{id:"\u5b89\u88c5\u4f9d\u8d56\u5305",children:"\u5b89\u88c5\u4f9d\u8d56\u5305"}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-shell",children:"pip3 install poetry     # \u4f7f\u7528poetry\u7ba1\u7406python\u5305\npoetry install          # \u5b89\u88c5\u4f9d\u8d56\n\npoetry shell            # \u8fdb\u5165\u865a\u62df\u73af\u5883\n"})}),"\n",(0,r.jsx)(t.h2,{id:"\u57fa\u7840\u914d\u7f6e",children:"\u57fa\u7840\u914d\u7f6e"}),"\n",(0,r.jsxs)(t.h4,{id:"\u8bbe\u7f6e\u8d85\u7ea7\u7528\u6237\u6253\u5f00-envdev-\u6587\u4ef6\u5728superusers\u548cqq\u4e2d\u6dfb\u52a0\u81ea\u5df1\u7684qq",children:["\u8bbe\u7f6e\u8d85\u7ea7\u7528\u6237\uff0c\u6253\u5f00 ",(0,r.jsx)(t.strong,{children:".env.dev"})," \u6587\u4ef6\uff0c\u5728",(0,r.jsx)(t.code,{children:"SUPERUSERS"}),"\u548c",(0,r.jsx)(t.code,{children:"qq"}),"\u4e2d\u6dfb\u52a0\u81ea\u5df1\u7684QQ"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="env.dev"',children:'SUPERUSERS=["123456789"]\n\nPLATFORM_SUPERUSERS = \'\n  {\n    "qq": ["123456789"],\n    "dodo": [],\n    "kaiheila": [],\n    "discord": []\n  }\n\'\n'})}),"\n",(0,r.jsx)(t.h4,{id:"\u6570\u636e\u5e93\u914d\u7f6e",children:"\u6570\u636e\u5e93\u914d\u7f6e"}),"\n",(0,r.jsxs)(q,{children:[(0,r.jsxs)(_,{value:"Postgresql",label:"Postgresql",default:!0,children:[(0,r.jsxs)(t.p,{children:["\u5efa\u8bae\u7684\u6570\u636e\u5e93\uff0c\u5acc\u9ebb\u70e6\u8bf7\u4f7f\u7528",(0,r.jsx)(t.code,{children:"Sqlite"})]}),(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="env.dev"',children:'# \u793a\u4f8b: "postgres://user:password@127.0.0.1:5432/database"\n\nDB_URL = "postgres://\u7528\u6237\u540d:\u5bc6\u7801@127.0.0.1:\u7aef\u53e3/\u6570\u636e\u5e93\u540d\u79f0"\n\n# \u5982\u679c\u4f60\u662f\u4e0e\u6559\u7a0b\u4e00\u6a21\u4e00\u6837\u7684\u547d\u4ee4\u4ee3\u7801\uff0c\u4e14\u6570\u636e\u5e93\u4e5f\u5728\u8be5\u670d\u52a1\u5668\u4e0a\uff0c\u53ef\u4ee5\u76f4\u63a5\u590d\u5236\u4ee5\u4e0bURL\nDB_URL = "postgres://postgres:zhenxun_bot@127.0.0.1:5432/zhenxun_bot"\n'})})]}),(0,r.jsxs)(_,{value:"Sqlite",label:"Sqlite",children:[(0,r.jsxs)(t.p,{children:["Sqlite\u53ef\u4ee5\u653e\u7f6e\u5728\u4efb\u4f55\u4f4d\u7f6e\uff0c",(0,r.jsx)(t.code,{children:"sqlite:"}),"\u4e4b\u540e\u4e3a\u5b58\u653e\u8def\u5f84\uff0c\u4f46\u5efa\u8bae\u5728data\u6587\u4ef6\u5939\u4e0b\u65b0\u5efadb\u6587\u4ef6\u5939\u540e\u4f7f\u7528\u4ee5\u4e0b"]}),(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="env.dev"',children:'# \u793a\u4f8b: "sqlite:data/db/zhenxun.db"\n\nDB_URL = "sqlite:data/db/zhenxun.db"\n'})})]}),(0,r.jsxs)(_,{value:"Mysql",label:"Mysql",children:[(0,r.jsxs)(t.p,{children:["\u4e0e",(0,r.jsx)(t.code,{children:"Postgresql"}),"\u76f8\u540c"]}),(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="env.dev"',children:'# \u793a\u4f8b: "mysql://user:password@127.0.0.1:3306/database"\n\nDB_URL = "mysql://\u7528\u6237\u540d:\u5bc6\u7801@127.0.0.1:\u7aef\u53e3/\u6570\u636e\u5e93\u540d\u79f0"\n'})})]})]}),"\n",(0,r.jsx)(t.h4,{id:"\u57fa\u7840\u63d2\u4ef6\u914d\u7f6e",children:"\u57fa\u7840\u63d2\u4ef6\u914d\u7f6e"}),"\n",(0,r.jsxs)(t.p,{children:["\u6587\u4ef6\u4fdd\u5b58\u5728 ",(0,r.jsx)(t.strong,{children:"data/config.yaml"}),"\uff0c\u6240\u6709\u771f\u5bfb\u76f8\u5173\u63d2\u4ef6\u90fd\u5728\u4f7f\u7528\u8be5\u914d\u7f6e\u6587\u4ef6\uff0c\u6309\u9700\u4fee\u6539"]}),"\n",(0,r.jsx)(t.h2,{id:"\u542f\u52a8",children:"\u542f\u52a8"}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-bash",children:"python3 bot.py\n# or\npython bot.py\n"})}),"\n",(0,r.jsx)(t.h4,{id:"\u5f53\u4f60\u7684\u63a7\u5236\u53f0\u51fa\u73b0\u4ee5\u4e0b\u65e5\u5fd7\u8bf4\u660e\u4f60\u5df2\u7ecf\u6210\u529f\u4e86",children:"\u5f53\u4f60\u7684\u63a7\u5236\u53f0\u51fa\u73b0\u4ee5\u4e0b\u65e5\u5fd7\uff0c\u8bf4\u660e\u4f60\u5df2\u7ecf\u6210\u529f\u4e86\ud83c\udf89\ud83c\udf89"}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-bash",children:"08-14 23:18:44 [INFO] zhenxun | CMD[Web UI] API\u542f\u52a8\u6210\u529f\n08-14 23:18:44 [INFO] uvicorn | Application startup complete.\n08-14 23:18:44 [INFO] uvicorn | Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)\n"})}),"\n",(0,r.jsx)(t.admonition,{title:"\u63d0\u793a",type:"tip",children:(0,r.jsx)(t.p,{children:"\u5f53\u524d\u7248\u672c\u771f\u5bfb\u672c\u4f53\u4e0e\u63d2\u4ef6\u5e93\u5206\u79bb\uff0c\u4f60\u53ef\u4ee5\u5728\u4ee5\u4e0b\u83b7\u53d6\u63d2\u4ef6\u6216\u5176\u4ed6\u76f8\u5173\uff0c\u6216\u901a\u8fc7\u4e0e\u771f\u5bfb\u7684\u5bf9\u8bdd\u547d\u4ee4\u5b89\u88c5\u63d2\u4ef6\uff08\u63d2\u4ef6\u5546\u5e97\uff09"})}),"\n",(0,r.jsxs)(t.table,{children:[(0,r.jsx)(t.thead,{children:(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.th,{style:{textAlign:"center"},children:"\u9879\u76ee\u540d\u79f0"}),(0,r.jsx)(t.th,{style:{textAlign:"center"},children:"\u4e3b\u8981\u7528\u9014"}),(0,r.jsx)(t.th,{style:{textAlign:"center"},children:"\u4ed3\u5e93\u4f5c\u8005"}),(0,r.jsx)(t.th,{style:{textAlign:"center"},children:"\u5907\u6ce8"})]})}),(0,r.jsxs)(t.tbody,{children:[(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/zhenxun-org/zhenxun_bot_plugins",children:"\u63d2\u4ef6\u5e93"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u63d2\u4ef6"}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/zhenxun-org",children:"zhenxun-org"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u539f plugins \u6587\u4ef6\u5939\u63d2\u4ef6"})]}),(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/zhenxun-org/zhenxun_bot_plugins_index",children:"\u63d2\u4ef6\u7d22\u5f15\u5e93"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u63d2\u4ef6"}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/zhenxun-org",children:"zhenxun-org"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u6269\u5c55\u63d2\u4ef6\u7d22\u5f15\u5e93"})]}),(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/soloxiaoye2022/zhenxun_bot-deploy",children:"\u4e00\u952e\u5b89\u88c5"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u5b89\u88c5"}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/soloxiaoye2022",children:"soloxiaoye2022"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u7b2c\u4e09\u65b9"})]}),(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/HibiKier/zhenxun_bot_webui",children:"WebUi"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u7ba1\u7406"}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/HibiKier",children:"hibikier"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u57fa\u4e8e\u771f\u5bfb WebApi \u7684 webui \u5b9e\u73b0"})]}),(0,r.jsxs)(t.tr,{children:[(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/YuS1aN/zhenxun_bot_android_ui",children:"\u5b89\u5353 app(WebUi)"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u5b89\u88c5"}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:(0,r.jsx)(t.a,{href:"https://github.com/YuS1aN",children:"YuS1aN"})}),(0,r.jsx)(t.td,{style:{textAlign:"center"},children:"\u7b2c\u4e09\u65b9"})]})]})]})]})}function E(e={}){const{wrapper:t}={...(0,l.R)(),...e.components};return t?(0,r.jsx)(t,{...e,children:(0,r.jsx)(k,{...e})}):k(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>i,x:()=>a});var r=n(6540);const l={},s=r.createContext(l);function i(e){const t=r.useContext(s);return r.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function a(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(l):e.components||l:i(e.components),r.createElement(s.Provider,{value:t},e.children)}}}]);