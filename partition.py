from PIL import Image
import cv2
import numpy as np

# 分区对应
organs = {
    1: "胃",
    2: "脾",
    3: "大肠",
    4: "小肠",
    5: "心",
    6: "乳",
    7: "肺",
    8: "肾",
    9: "膀胱",
    10: "肾",
    11: "女子胞（男子为外肾）",
    12: "肝",
    13: "胆",
    14: "肝",
    15: "脾",
    16: "脑、骨之颈臂手及相应之髓部",
    17: "骨之腰骶腿足及相应之髓部"
}


# 等比缩放
def resize_image(image_path, max_size=1024):
    image = Image.open(image_path)
    width, height = image.size

    # 计算缩放比例
    scale = max_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))

    resized_image = image.resize(new_size)
    resized_path = "resized_temp.jpg"
    resized_image.save(resized_path)
    return resized_path


def map_region_to_position(region_center, xr, yr, rr, xg, yg, rg, is_right_eye=True):
    """
    根据区域中心点的位置确定区域编号
    params:
        region_center: 区域中心点坐标 (x, y)
        xr, yr, rr: 虹膜圆的中心x,y坐标和半径
        xg, yg, rg: 瞳孔圆的中心x,y坐标和半径
        is_right_eye: 是否为右眼，True为右眼，False为左眼
    """
    x, y = region_center

    # 判断是否在瞳孔左右切线之间
    in_pupil_vertical = xg - rg <= x <= xg + rg
    # 判断是否在虹膜左右切线之间
    in_iris_vertical = xr - rr <= x <= xr + rr
    # 判断是否在虹膜圆上方或下方
    above_iris_center = y < yr

    # 瞳孔切线内的区域
    if in_pupil_vertical:
        if above_iris_center:
            return 9  # 膀胱
        else:
            return 1  # 胃

    # 虹膜切线内、瞳孔切线外的区域
    elif in_iris_vertical:
        if above_iris_center:
            if x < xr:  # 左侧
                return 10 if is_right_eye else 8  # 肾（对调）
            else:  # 右侧
                return 8 if is_right_eye else 10  # 肾（对调）
        else:
            if x < xr:  # 左侧
                return 15 if is_right_eye else 2  # 脾（对调）
            else:  # 右侧
                return 2 if is_right_eye else 15  # 脾（对调）

    # 虹膜切线外的区域
    else:
        # 检查是否在瞳孔的上下切线之间
        in_pupil_horizontal = yg - rg <= y <= yg + rg
        if in_pupil_horizontal:
            # 瞳孔上下切线之间的区域
            if x < xr - rr:  # 左侧
                return 13 if is_right_eye else 5  # 胆/心（对调）
            elif x > xr + rr:  # 右侧
                return 5 if is_right_eye else 13  # 心/胆（对调）
        elif y < yg - rg:  # 瞳孔上切线以上
            if x < xr - rr:  # 左侧
                return 11 if is_right_eye else 6  # 女子胞/乳（对调）
            elif x > xr + rr:  # 右侧
                return 6 if is_right_eye else 11  # 乳/女子胞（对调）
        elif y > yg + rg:  # 瞳孔下切线以下
            if x < xr - rr:  # 左侧
                return 14 if is_right_eye else 3  # 肝/大肠（对调）
            elif x > xr + rr:  # 右侧
                return 3 if is_right_eye else 14  # 大肠/肝（对调）

    return None  # 其他区域暂不处理


def draw_region(image_path, point_x=None, point_y=None, is_right_eye=True):
    resized_path = resize_image(image_path)
    image = cv2.imread(resized_path)
    height, width, _ = image.shape

    # 提取颜色通道
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # 生成掩码
    _, blue_mask = cv2.threshold(blue_channel, 127, 255, cv2.THRESH_BINARY)
    _, green_mask = cv2.threshold(green_channel, 127, 255, cv2.THRESH_BINARY)
    _, red_mask = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)

    # 查找所有轮廓
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建巩膜区域掩码（蓝色区域减去红色圆形区域）
    sclera_mask = np.zeros((height, width), dtype=np.uint8)

    # 填充蓝色区域
    cv2.drawContours(sclera_mask, blue_contours, -1, 255, -1)

    # 减去红色虹膜区域
    if red_contours:
        largest_red = max(red_contours, key=cv2.contourArea)
        (xr, yr), rr = cv2.minEnclosingCircle(largest_red)
        xr, yr, rr = int(xr), int(yr), int(rr)
        center = (int(xr), int(yr))
        radius = int(rr)
        cv2.circle(sclera_mask, center, radius, 0, -1)

    # 提取巩膜区域作为输出图像
    output_image = cv2.bitwise_and(image, image, mask=sclera_mask)

    def draw_limited_line(start, end):
        # 创建一个空白掩码
        mask = np.zeros_like(blue_mask)

        # 绘制完整的线段
        cv2.line(mask, start, end, 255, 2)

        # 将线段限制在巩膜区域内
        limited_mask = cv2.bitwise_and(mask, sclera_mask)
        # 使用黑色 (0,0,0) 绘制线条
        output_image[limited_mask == 255] = (0, 0, 0)

    # 存储红色虹膜的边界位置
    iris_boundaries = []

    # 处理红色区域（虹膜）- 只处理最大区域
    if red_contours:
        # 找到最大的红色轮廓
        largest_red = max(red_contours, key=cv2.contourArea)
        (xr, yr), rr = cv2.minEnclosingCircle(largest_red)
        xr, yr, rr = int(xr), int(yr), int(rr)
        center = (int(xr), int(yr))
        radius = int(rr)

        # 使用黑色绘制圆
        cv2.circle(output_image, center, radius, (0, 0, 0), 2)

        # 存储最大虹膜的左右边界
        iris_boundaries.append({
            'left': xr - rr,
            'right': xr + rr,
            'center_y': yr
        })

        # 绘制切线
        draw_limited_line((xr - rr, 0), (xr - rr, height))  # 左切线
        draw_limited_line((xr + rr, 0), (xr + rr, height))  # 右切线

    # 处理绿色区域（瞳孔）- 只处理最大区域
    if green_contours and iris_boundaries:
        # 找到最大的绿色轮廓
        largest_green = max(green_contours, key=cv2.contourArea)
        (xg, yg), rg = cv2.minEnclosingCircle(largest_green)
        xg, yg, rg = int(xg), int(yg), int(rg)

        iris = iris_boundaries[0]  # 使用最大虹膜的边界

        # 计算绿色切线与红色圆的交点
        def get_circle_line_intersection(x, y_center, r):
            # 计算直线x = const与圆的交点
            if abs(x - xr) > rr:  # 如果直线在圆外
                return None, None

            # 计算交点的y坐标
            delta = rr ** 2 - (x - xr) ** 2
            if delta < 0:
                return None, None

            y1 = yr - np.sqrt(delta)
            y2 = yr + np.sqrt(delta)
            return y1, y2

        # 左切线的交点
        y1_left, y2_left = get_circle_line_intersection(xg - rg, yr, rr)
        # 右切线的交点
        y1_right, y2_right = get_circle_line_intersection(xg + rg, yr, rr)

        # 绘制竖直切线
        # 左切线
        if y1_left is not None:
            if 0 < y1_left:
                draw_limited_line((xg - rg, 0), (xg - rg, int(y1_left)))
            if y2_left < height:
                draw_limited_line((xg - rg, int(y2_left)), (xg - rg, height))
        else:
            # 如果没有交点，画整条线
            draw_limited_line((xg - rg, 0), (xg - rg, height))

        # 右切线
        if y1_right is not None:
            if 0 < y1_right:
                draw_limited_line((xg + rg, 0), (xg + rg, int(y1_right)))
            if y2_right < height:
                draw_limited_line((xg + rg, int(y2_right)), (xg + rg, height))
        else:
            # 如果没有交点，画整条线
            draw_limited_line((xg + rg, 0), (xg + rg, height))

        # 绘制水平切线，只画虹膜边界外的部分
        # 上切线左段
        if 0 < yg - rg:
            draw_limited_line((0, yg - rg), (iris['left'], yg - rg))
        # 上切线右段
        if iris['right'] < width:
            draw_limited_line((iris['right'], yg - rg), (width, yg - rg))

        # 下切线左段
        if 0 < yg + rg:
            draw_limited_line((0, yg + rg), (iris['left'], yg + rg))
        # 下切线右段
        if iris['right'] < width:
            draw_limited_line((iris['right'], yg + rg), (width, yg + rg))

    # 在显示图像之前添加统计代码
    # 创建掩码来识别蓝色区域（非黑色区域）
    blue_regions_mask = np.zeros((height, width), dtype=np.uint8)
    # 将所有非黑色像素标记为255（蓝色区域）
    blue_regions_mask[np.any(output_image > 0, axis=2)] = 255

    # 使用形态学操作来清理可能的噪点
    kernel = np.ones((3, 3), np.uint8)
    blue_regions_mask = cv2.morphologyEx(blue_regions_mask, cv2.MORPH_CLOSE, kernel)

    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_regions_mask)

    # 过滤掉太小的区域（可能是噪点）
    min_area = 100
    valid_regions = 0
    region_mapping = {}  # 用于存储原始标签到实际区域编号的映射
    valid_labels = set()  # 存储有效区域的标签

    # 首先处理和标记有效区域
    for i in range(1, num_labels):  # 从1开始，跳过背景
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            # 获取区域中心点
            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])

            # 获取区域的实际编号（根据位置）
            actual_region = map_region_to_position(
                (center_x, center_y),
                xr, yr, rr,  # 虹膜参数
                xg, yg, rg,  # 瞳孔参数
                is_right_eye  # 添加左右眼参数
            )

            # 如果能够确定区域编号，则保存映射关系
            if actual_region is not None:
                valid_regions += 1
                region_mapping[i] = actual_region
                valid_labels.add(i)

    # 在可视化阶段，处理所有非背景区域
    for i in range(1, num_labels):
        # 将所有非背景区域填充为白色
        output_image[labels == i] = [255, 255, 255]
        
        # 只在有效区域上添加编号
        if i in valid_labels:
            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])
            cv2.putText(output_image, str(region_mapping[i]), (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print(f"图像中共有 {valid_regions} 个有效区域")

    # 如果提供了坐标点，判断点所在的区域
    if point_x is not None and point_y is not None:
        # 确保点在图像范围内
        if 0 <= point_x < width and 0 <= point_y < height:
            # 获取点所在区域的原始标签
            region_label = labels[point_y, point_x]
            if region_label > 0 and region_label in region_mapping:  # 0是背景
                actual_region = region_mapping[region_label]
                print(f"坐标点 ({point_x}, {point_y}) 位于区域 {actual_region}, 该区域对应{organs[actual_region]}")
                # 在图像上标记该点
                cv2.circle(output_image, (point_x, point_y), 3, (0, 0, 255), -1)
            else:
                # 点不在任何区域内，计算到最近区域的距离
                min_distance = float('inf')
                nearest_region = None

                # 遍历所有有效区域
                for i in range(1, num_labels):
                    if i in region_mapping:
                        # 创建当前区域的掩码
                        region_mask = (labels == i).astype(np.uint8)

                        # 计算点到区域的最小距离
                        dist = cv2.pointPolygonTest(
                            cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0],
                            (point_x, point_y),
                            True  # 返回实际距离而不是内外测试
                        )

                        # 距离的绝对值（因为pointPolygonTest对区域外的点返回负值）
                        dist = abs(dist)

                        if dist < min_distance:
                            min_distance = dist
                            nearest_region = region_mapping[i]

                print(f"坐标点 ({point_x}, {point_y}) 不在任何区域内")
                if nearest_region is not None:
                    print(f"最近的区域是区域 {nearest_region}，距离约为 {min_distance:.2f} 像素")

                # 在图像上标记该点
                cv2.circle(output_image, (point_x, point_y), 3, (0, 0, 255), -1)

                # 可选：绘制一条线连接点和最近区域的中心
                if nearest_region is not None:
                    # 找到最近区域的中心点
                    for i, mapped_region in region_mapping.items():
                        if mapped_region == nearest_region:
                            center_x = int(centroids[i][0])
                            center_y = int(centroids[i][1])
                            # 绘制虚线连接点和区域中心
                            cv2.line(output_image, (point_x, point_y), (center_x, center_y),
                                     (0, 0, 255), 1, cv2.LINE_AA)
                            break
        else:
            print("坐标点超出图像范围")

    # 显示巩膜区域和切线
    cv2.imshow("Sclera Region with Tangent Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img_path = 'result/result/042.jpg'
    draw_region(img_path, point_x=200, point_y=350, is_right_eye=False)


if __name__ == '__main__':
    main()
