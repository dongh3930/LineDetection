import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

#색 강조
def color_emphasis(img):
  img[:, :,0] = img[:, :,0] * 0.1
  img[:, :,1] = img[:, :,1] * 0.8
  img[:, :,2] = img[:, :,2] * 1
  return img

#Color Filter (흰색 노란색 강조)
def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV 검출조건 색상/채도/명도
    white_lower = np.array([0, 50, 0])
    white_upper = np.array([255, 255, 255])
    yellow_lower = np.array([20, 50, 100])
    yellow_upper = np.array([60, 255, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(img, img, mask = mask)
    return masked

#Grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Gaussian Filter (노이즈 제거)
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    #Sigma 0설정시 커널 크기를 고려하여 자동으로 설정함

#모폴로지 연산 (닫힘연산: 팽창 + 침식 (주변보다 어두운 도이즈 제거 (끊어진 곳 연결 및 구멍 메꿈)))
def mophology(img):
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  mop_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  return mop_img

#Sharp Filter
def sharp_filter(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_img = cv2.filter2D(img, -1, kernel)
    return sharp_img

#Canny Edge Detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

#ROI
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2: #컬러 이미지일 때
        channel_count = img.shape[2] #이미지 색상 채널 수
        ignore_mask_color = (255,) * channel_count
    else: #흑백 이미지일 때
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color) #다각형 그리기 함수(타겟이미지, 포인트 데이터(2차원 배열),색상)
    ROI_img = cv2.bitwise_and(img, mask) #비트 연산
    return ROI_img

#Hough Transformation
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    height, width = img.shape[:2]
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    if (lines is None): #검출 못했을 때 중간 값으로 반환
        return np.array([[[int(width / 2), int(height), int(width / 2), int(height / 2)]]])
    else:
        return lines

#이미지 합성 (가중치 합)
def weighted_img(img, initial_img, a, b, c):
    return cv2.addWeighted(initial_img, a, img, b, c) #영상1, 가중치1, 영상2, 가중치2, 결과영상에 추가할 값

'''
#Draw Lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
'''

#대표선 구하기
def get_fitline(img, f_lines):
    #lines = np.squeeze(f_lines) #갯수, 좌표(2)
    #lines = lines.reshape(lines.shape[0] * 2, 2) #갯수x2, 좌표(1) 로 변환
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01) #좌표, 방법(최소제곱법), 매개변수, 정확도(반경,각도)
    vx, vy, x, y = output[0], output[1], output[2], output[3] #정규화된 벡터, 선 위에 한 점
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2]
    return result

#대표선 그리기
def draw_fitline(img, result_l,result_r, middle_line, color=(255,0,255), thickness = 10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]) , int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]) , int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    cv2.line(lane, (int(middle_line[0]), int(middle_line[1])), (int(middle_line[2]), int(middle_line[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img, 0.8, 1, 0)
    return final

#검출한 모든 직선 내 점들 수집
def Collect_points(lines):
    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0] * 2, 2) #갯수x2, 좌표(1) 로 변환
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0]
            b = line[1]
            c = line[2]
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1]) #기울기
            for m in range(0, tmp, 5): #start / stop / step
                if slope > 0: #양의 기울기
                    new_point = np.array([[int(a + m * slope), int(b + m)]])
                    interp = np.concatenate((interp, new_point), axis=0) #직선 위 새로운 점들 추가
                elif slope < 0: #음의 기울기
                    new_point = np.array([[int(a - m * slope), int(b - m)]]) #직선 위 새로운 점들 추가
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp

#랜덤으로 두 점을 뽑기
def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if (two[0] == one[0]):  # extract again if values are overlapped
        while two[0] == one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1, 2), two.reshape(1, 2)
    three = np.concatenate((one, two), axis=1) #두 배열 결합 (1x4)
    three = three.squeeze()
    return three

#모델 parameter 계산
def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1]) / (line[2] - line[0]) #기울기
    n = line[1] - m * line[0] #절편
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a, b, c])
    return par

#점과 직선 사이의 거리 계산
def compute_distance(par, point):
    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)

#모든 점들과 랜덤 직선 사이의 거리의 평균 오차
def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par, lines)
    # total sum of distance between random line and sample points
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist / len(lines)

    return avg_dist

#불필요한 직선 제거
def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par, lines)

    # filtered_dist = distance[distance<15]
    filtered_lines = lines[distance < 13, :]
    return filtered_lines

#RANSAC 알고리즘
def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0])
    if (len(lines) != 0):
        for i in range(len(lines)):
            sample = get_random_samples(lines) #L_interp, R_interp 점들 중 임의로 2점 뽑음(좌표)
            parameter = compute_model_parameter(sample) #뽑은 두 점에 대한 직선 방정식 parameter 계산
            cost = model_verification(parameter, lines) #오차
            if cost < min:  # update best_line (오차 값이 작으면)
                min = cost
                best_line = parameter #최적의 직선 파라미터
            if min < 3: break #오차 값이 3보다 작으면 탈출

        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        #대표선 그리기
        fit_result = get_fitline(img, filtered_lines)

        #기울기가 음수면 왼쪽 차선 / 양수면 오른쪽 차선
        if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result
    else: #검출 못했을 때 중간 값으로 반환
        return [int(width/2), int(height), int(width/2), int(height/2)]

#Smoothing(프레임 부드럽게)
def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0, 0, 0, 0])

    for ii, line in enumerate(reversed(lines)): #인덱스와 값 모두 추출
        #매 10프레임마다 평균치를 출력하여 부드러운 움직이게함
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

#직선 사이 거리 측정을 위한 x값 구하기
def compute_model_line(line, y):
    m = (line[3] - line[1]) / (line[2] - line[0]) #기울기
    n = line[1] - m * line[0]  # 절편
    x = np.array([(1 / m) * y - (n / m)])
    return x

#차선 검출
def detect_lanes_img(img, y):
    height, width = img.shape[:2]

    vertices = np.array([[(50, height), ((width / 2) - 100, (height / 2) + 50), ((width / 2) + 100, (height / 2) + 50), ((width -50), height)]], dtype=np.int32)
    ROI_image = region_of_interest(img, vertices)

    color_emp_image = color_emphasis(ROI_image)
    color_filtered_image = color_filter(color_emp_image)

    gray_image = grayscale(color_filtered_image)

    kernel_size = 3  # 가우시안 필터크기
    blur_image = gaussian_blur(gray_image, kernel_size)
    mop_image = mophology(blur_image)
    sharped_image = sharp_filter(mop_image)

    low_threshold = 70  # 하위 임계값 (낮으면 고려하지 않음)
    high_threshold = 210  # 상위 임계값 (크면 엣지로 검출)
    cannyed_image = canny(sharped_image, low_threshold, high_threshold)

    rho = 1
    theta = 1 * np.pi / 180
    threshold = 50
    min_line_len = 50  # 선의 최소 길이
    max_line_gap = 150  # 선 사이의 최대 허용 간격
    line_arr = hough_lines(cannyed_image, rho, theta, threshold, min_line_len, max_line_gap)
    #line_arr = np.squeeze(line_arr)  # 크기 1인 axis 제거

    # Get slope degree to separate 2 group (+ slope , - slope)
    slope_degree = (np.arctan2(line_arr[:, :, 1] - line_arr[:, :, 3], line_arr[:, :, 0] - line_arr[:, :, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # ignore vertical slope lines
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]

    # interpolation & collecting points for RANSAC
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    # erase outliers based on best line
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    # smoothing by using previous frames (매 프레임마다 최적의 차선 저장)
    L_lane.append(left_fit_line), R_lane.append(right_fit_line)

    if len(L_lane) > 10:
        left_fit_line = smoothing(L_lane, 10)
    if len(R_lane) > 10:
        right_fit_line = smoothing(R_lane, 10)

    left_x = compute_model_line(left_fit_line, y)
    right_x = compute_model_line(right_fit_line, y)
    middle_line = [left_x, y, right_x, y]

    final = draw_fitline(img, left_fit_line, right_fit_line, middle_line)

    return final

fit_result, l_fit_result, r_fit_result, L_lane, R_lane, middle_line = [], [], [], [], [], []

cap = cv2.VideoCapture('challenge.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    cropped_image = frame[0:height - 35, 0:width].copy()  # 이미지 특정 구역 잘라내기
    #if frame.shape[0] !=540: # resizing for challenge video (영상 축소)
    #    frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA)
    result = detect_lanes_img(cropped_image, y=600)

    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()