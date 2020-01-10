## Android Touch 事件传递机制
Android 事件管理机制是一名专业 Android 研发工程师必备知识。
### 传递方式
从父布局开始一步一步的向下子布局分发事件。但是如果某一个
分发事件调用 dispatchTouchEvent

#### 应用场景
任务分发，分派给开发经纪

Touch 事件传递时，每次分发之后，会调用拦截方法 boolean

Touch 事件传递拥有记忆功能，处理了
- 
