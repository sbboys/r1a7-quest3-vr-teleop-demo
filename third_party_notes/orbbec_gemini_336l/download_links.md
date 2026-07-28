# Gemini 336L 官方开发资料下载入口

设备型号：Orbbec Gemini 336L，属于 Gemini 330 系列双目结构光相机。

## 官网资料页

- Gemini 336L 所在资料下载页：https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1
- Gemini 336L 产品页：https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=65
- Gemini 330 系列中文文档主页：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=74

## 必需下载

当前工程在 Ubuntu x86_64 上运行，优先下载 Linux x86_64 版本。

- OrbbecSDK v2 releases：https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases/
- 最新 SDK Linux x86_64 tar.gz：
  https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases/download/v2.8.7/OrbbecSDK_v2.8.7_202606161335_ab8672c_linux_x86_64.tar.gz
- 最新 Orbbec Viewer Linux x86_64 tar.gz：
  https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases/download/v2.8.7/OrbbecViewer_v2.8.7_202606161335_71becaa_linux_x86_64.tar.gz

## 备用安装包

如果 tar.gz 不方便使用，可下载上一版 deb：

- OrbbecSDK v2.8.6 amd64 deb：
  https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases/download/v2.8.6/OrbbecSDK_v2.8.6_amd64.deb

## 必读文档

- 安装 Orbbec SDK：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=104
- 支持的上位机和系统要求：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=176
- 相机简易使用指南：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=79
- 通过 Orbbec Viewer 试用相机功能：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=80
- 深度到彩色空间对齐：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=88
- 坐标系转换：https://www.orbbec.com.cn/index/Gemini330/info.html?cate=119&id=114

## 与本工程的关系

1. 先安装 OrbbecSDK v2。
2. 用 Orbbec Viewer 确认 Gemini 336L 可以输出 color/depth。
3. 再让 `pyorbbecsdk` 可被 Python 导入。
4. 最后运行本工程的 `tools/test_gemini_pose.py` 和 `sim_main.py --action_source camera_pose`。
