import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img,sin_img,cos_img):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(3, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(3,2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(3, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=0, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(3, 2, 5)
    plot = ax.imshow(sin_img, cmap='hsv', vmin=0, vmax=1)
    ax.set_title('Sin')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(3, 2, 6)
    plot = ax.imshow(cos_img, cmap='hsv', vmin=0, vmax=1)
    ax.set_title('Cos')
    ax.axis('off')
    plt.colorbar(plot)

    plt.show()

def detect_grasps(q_img,no_grasps=5):
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    print("Here is the max value point:")
    for grasp_point_array in local_max:
        print(grasp_point_array)

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_q,ground_truth_a):
    #先判斷重合區域IOU大於25% 否則 return False
    ground_truth_q = ground_truth_q.cpu().detach().numpy().reshape((600, 600))
    ground_truth_a = ground_truth_a.cpu().detach().numpy().reshape((600, 600))
    IOU_q = np.array(np.where(ground_truth_q * grasp_q > 0.2))
    #print(IOU_q)
    #detect_grasps(grasp_q,5)
    a = IOU_q.shape[1]
    GTQ = np.array(np.where(ground_truth_q > 0.9))
    b = GTQ.shape[1]
    #print(a,"-",b,"-",a/b)
    AngleSUM = 0
    GT_AngleSUM =0
    try:
        if a / b < 0.25:
            return False
        else:
            # 再判斷重合區域的angle_threshold=np.pi/6 否則 return False
            for i in range(a):
                AngleSUM = grasp_angle[IOU_q[0][i]][IOU_q[1][i]] + AngleSUM
                GT_AngleSUM = ground_truth_a[IOU_q[0][i]][IOU_q[1][i]]+GT_AngleSUM
            if abs(AngleSUM - GT_AngleSUM)/a > np.pi / 6:
                return False
            else:
                return True
    except Exception as e:
        print(str(e))
        return False