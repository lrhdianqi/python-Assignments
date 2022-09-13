import cv2
# import tracker2 as tracker

cap = cv2.VideoCapture('./video/test.mp4')

tracker = cv2.TrackerKCF_create()

#############

# 在视频的第一帧中，通过手动绘制boundingbox去抓取目标，抓取成功之后敲击回车键，boundingbox便会被视为之后将要tracking的对象
success, frame = cap.read()

boundingBox = cv2.selectROI("Tracking", frame, False)

frame = cv2.resize(frame, (960, 540))
tracker.init(frame, boundingBox)

# 在接下来的视频的其他帧中，按照之前第一帧自己绘制的bounding box的顶点、长、宽去重复添加boundingbox
def drawBox(img, boundingBox):
    x, y, w, h = int(boundingBox[0]), int(boundingBox[1]), int(boundingBox[2]), int(boundingBox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 捕获成功地话在界面中输出：tracking


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, boundingBox = tracker.update(img)

    if success:  # 如果在第一帧手动添加boundingbox 成功的话
        drawBox(img, boundingBox)  # 绘制bounding box在接下来的帧中
    else:  # 如果第一帧手动添加boundingbox失败的话则在界面上输出：lost the target
        cv2.putText(img, "Lost the target", (75, 75), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)

    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)  # 显示fps
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  # 获取视频的fps
    # 根据视频fps的大小去定义显示fps字体的颜色
    if fps > 60:
        myColor = (20, 230, 20)
    elif fps > 20:
        myColor = (230, 20, 20)
    else:
        myColor = (20, 20, 230)
    cv2.putText(img, "FPS:" + str(int(fps)), (75, 50), cv2.FONT_ITALIC, 0.9, myColor, 2)  # 显示fps

    cv2.imshow("Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
