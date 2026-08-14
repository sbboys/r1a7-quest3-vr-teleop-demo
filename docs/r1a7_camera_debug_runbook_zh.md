# R1-A7 相机调试记录与操作手册

本文记录 2026-08-13 至 2026-08-14 对 R1-A7 头部相机和双臂腕部相机的现场调试结论。重点结论是：头部相机当前未通过 AP/RTSP/UDP 对电脑开放视频流；双臂腕部相机可作为 USB UVC 摄像头直连上位机查看。

## 硬件与网络

电脑侧已验证的网络和设备：

- 机器人控制网口：`enx9c69d37d0967`
- 控制网段电脑 IP：`192.168.123.223/24`
- 机器人控制 IP：`192.168.123.161`
- 机器人 AP SSID：`R1_05062_d82b4296`
- 机器人 AP 网关：`192.168.12.1`
- 电脑连接机器人 AP 后的 IP：`192.168.12.78`
- USB 腕部相机 Hub：`1a86:809f QinHeng Electronics USB2.0 HUB`
- 腕部相机视频节点：
  - `/dev/video0`：`JR0001`
  - `/dev/video2`：`JR0002`

## 头部相机排查结论

宇树 R1-A 视频流服务文档说明头部双目模组可输出：

- `/dev/video-img`：1280x720，对应 UDP RTP H264 `5001`
- `/dev/video-left`：544x448，对应 UDP RTP H264 `5002`
- `/dev/video-right`：544x448，对应 UDP RTP H264 `5003`
- `/dev/video-dep`：544x448 深度图，需在机器人侧使用本地 V4L2 设备

文档同时说明：

- `video_hub` 可能与推流服务冲突；
- `Stereo patch PC1` 默认不自启动；
- 需要将双目视频推送到 NX 扩展坞侧时，手动开启 `Stereo patch PC1`。

现场条件：

- 机器人没有 NX 扩展坞；
- 电脑可以连接机器人 AP；
- 机器人 APP 中 `video_hub` 和 `videohub_rtsp_server` 可显示蓝色；
- 无法进入机器人本体/NX 终端配置服务。

已做检测：

```bash
ping -I wlxf8c9033cd8e2 -c 3 192.168.12.1
```

结果：机器人 AP 网关可达。

扫描常见 RTSP/HTTP/视频端口：

```bash
for p in 554 8554 5001 5002 5003 55555 55556 55557 60000 60001 60002 60003; do
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/192.168.12.1/$p" 2>/dev/null \
    && echo "OPEN $p" || echo "closed $p"
done
```

结果：常见 TCP 端口未开放。

抓取头部相机 UDP RTP 端口：

```bash
sudo timeout 20 tcpdump -ni wlxf8c9033cd8e2 \
  'host 192.168.12.1 and (udp port 5001 or udp port 5002 or udp port 5003)'
```

结果：`0 packets captured`。

抓取 AP 网络全部 UDP/TCP 流量：

```bash
sudo timeout 30 tcpdump -ni wlxf8c9033cd8e2 'udp or tcp'
```

结果：只看到局域网发现类 mDNS/LLMNR 包，没有相机大流量。

结论：

- 电脑和机器人 AP 网络正常；
- 当前没有发现头部相机通过 AP、RTSP、UDP RTP 或 `video_hub` 推送到电脑；
- 在无 NX 扩展坞且不能进入机器人端配置服务的条件下，电脑侧暂时不能直接获取头部相机视频。

## 腕部相机 USB 识别

腕部相机不需要通过机器人 APP 或 AP。按宇树说明，腕部相机可以作为 USB 设备直连上位机查看。

插入腕部相机 USB 后，电脑出现以下 USB 设备：

```text
1a86:809f QinHeng Electronics USB2.0 HUB
0001:0001 Fry's Electronics JR0001
0002:0002 Ingram passport00
1a86:5395 QinHeng Electronics USB 10/100 LAN
1a86:55e7 QinHeng Electronics UART+SPI+I2C+JTAG
1a86:80b5 QinHeng Electronics USB Multiple Card Reader_V1.0
```

其中两个视频设备挂载为 UVC：

```text
Port 3: Dev 11, If 0, Class=Video, Driver=uvcvideo
Port 3: Dev 11, If 1, Class=Video, Driver=uvcvideo
Port 4: Dev 12, If 0, Class=Video, Driver=uvcvideo
Port 4: Dev 12, If 1, Class=Video, Driver=uvcvideo
```

实际视频节点：

```bash
for v in /sys/class/video4linux/video*; do
  echo "===== $(basename "$v") ====="
  cat "$v/name" 2>/dev/null
  readlink -f "$v"
done
```

现场识别结果：

```text
video0 JR0001: JR0001
video1 JR0001: JR0001
video2 JR0002: JR0002
video3 JR0002: JR0002
```

GStreamer 采集测试：

```bash
for dev in /dev/video0 /dev/video1 /dev/video2 /dev/video3; do
  echo "===== $dev ====="
  timeout 5 gst-launch-1.0 -q v4l2src device=$dev num-buffers=5 ! fakesink 2>&1 \
    && echo OK || echo FAIL
done
```

结果：

- `/dev/video0`：可采集
- `/dev/video1`：不是采集设备
- `/dev/video2`：可采集
- `/dev/video3`：不是采集设备

## 腕部相机格式

两个可用采集节点均支持：

- MJPEG：最高 1920x1080，支持 60 fps 和 30 fps
- YUYV：640x480 可 30 fps；高分辨率帧率较低

双路同时打开时，不建议使用 YUYV 原始流。现场用 `YUY2 640x480@30` 同时打开两路时，第二路出现：

```text
Buffer pool activation failed
streaming stopped, reason not-negotiated (-4)
```

原因是双路 YUYV 原始流占用 USB 带宽和缓冲资源较高。改用 MJPEG 后可同时打开。

## 同时打开两个腕部相机

推荐使用 MJPEG：

终端 1：

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  image/jpeg,width=640,height=480,framerate=30/1 ! \
  jpegdec ! videoconvert ! autovideosink sync=false
```

终端 2：

```bash
gst-launch-1.0 v4l2src device=/dev/video2 ! \
  image/jpeg,width=640,height=480,framerate=30/1 ! \
  jpegdec ! videoconvert ! autovideosink sync=false
```

现场已验证：两路 MJPEG 640x480@30 可同时打开。

## 快速复查命令

插入腕部相机 USB 后执行：

```bash
lsusb
lsusb -t
ip -br addr
ls -l /dev/ttyACM*
ls -l /dev/video* 2>/dev/null || true
for v in /sys/class/video4linux/video*; do
  echo "===== $(basename "$v") ====="
  cat "$v/name" 2>/dev/null
  readlink -f "$v"
done
```

期望看到：

```text
/dev/video0  JR0001
/dev/video2  JR0002
Class=Video, Driver=uvcvideo
```

如果没有 `/dev/video*`：

1. 确认插入的是腕部相机 USB 数据线；
2. 保持机器人上电，腕部相机可能从机器人侧取电；
3. 换一根确认支持数据传输的 USB 线；
4. 换电脑后置 USB 口，避免经过不稳定 Hub；
5. 使用 `sudo dmesg -w` 观察插入瞬间是否出现 `New USB device found`。

## 最终结论

- 头部相机：当前未通过电脑可访问的 AP/RTSP/UDP 方式输出视频；无 NX 扩展坞时暂未拿到视频。
- 腕部相机：已确认可通过 USB 直连电脑作为 UVC 摄像头使用。
- 腕部相机可用节点：`/dev/video0` 和 `/dev/video2`。
- 双路同时预览应使用 MJPEG，不要用双路 YUYV 原始流。
