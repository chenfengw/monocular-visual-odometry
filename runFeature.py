import os
import numpy as np
import sys
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
sys.path.append("src")
import viso2
import matplotlib.pyplot as plt
from skimage.io import imread
import os.path as osp

def errorMetric(RPred, RGt, TPred, TGt):
    """Compute rotation and translation error

    Args:
        RPred (np array): rotation prediction. shape = (3,3)
        RGt (np array): rotation ground truth. shape = (3,3)
        TPred (np array): translation prediction. shape = (3,1)
        TGt (np array): translation ground truth. shape = (3,1)

    Returns:
        tuple: (errorRot(float), errorTrans(float))
    """
    diffRot = (RPred - RGt)
    diffTrans = (TPred - TGt)
    errorRot = np.sqrt(np.sum(np.multiply(diffRot.reshape(-1), diffRot.reshape(-1))))
    errorTrans = np.sqrt(np.sum(np.multiply(diffTrans.reshape(-1), diffTrans.reshape(-1))))

    return errorRot, errorTrans


def runSFM(dataset_path, feature_dir):
    if_vis = True # set to True to do the visualization per frame; the images will be saved at '.vis/'. Turn off if you just want the camera poses and errors
    if_on_screen = False # if True the visualization per frame is going to be displayed realtime on screen; if False there will be no display, but in both options the images will be saved

    # parameter settings (for an example, please download
    img_dir      = os.path.join(dataset_path, 'sequences/00/image_0')
    gt_dir       = os.path.join(dataset_path, 'poses/00.txt')
    calibFile    = os.path.join(dataset_path, 'sequences/00/calib.txt')
    suffix = 'feature'

    # Load the camera calibration information
    with open(calibFile) as fid:
        calibLines = fid.readlines()
        calibLines = [calibLine.strip() for calibLine in calibLines]

    calibInfo = [float(calibStr) for calibStr in calibLines[0].split(' ')[1:]]

    # Load the ground-truth depth and rotation
    with open(gt_dir) as fid:
        gtTr = [[float(TrStr) for TrStr in line.strip().split(' ')] for line in fid.readlines()]
    gtTr = np.asarray(gtTr).reshape(-1, 3, 4)

    first_frame  = 0
    last_frame   = 300

    # init visual odometry, set parameters for viso2
    params = viso2.Mono_parameters()
    params.calib.f = calibInfo[0]
    params.calib.cu = calibInfo[2]
    params.calib.cv = calibInfo[6]
    params.height = 1.6
    params.pitch = -0.08

    first_frame  = 0
    last_frame   = 300

    # init transformation matrix array
    Tr_total = []
    Tr_total_np = []
    Tr_total.append(viso2.Matrix_eye(4))
    Tr_total_np.append(np.eye(4))

    # init viso module
    visoMono = viso2.VisualOdometryMono(params)

    if if_vis:
        save_path = 'vis_preFeature'
        os.makedirs(save_path, exist_ok=True)

        # create figure
        fig = plt.figure(figsize=(10, 15))
        ax1 = plt.subplot(211) # plot features
        ax1.axis('off')
        ax2 = plt.subplot(212) # plot trajacotry
        ax2.set_xticks(np.arange(-100, 100, step=10))
        ax2.set_yticks(np.arange(-500, 500, step=10))
        ax2.axis('equal')
        ax2.grid()
        if if_on_screen:
            plt.ion()
            plt.show()
        else:
            plt.ioff()

    # for all frames do
    if_replace = False
    errorTransSum = 0
    errorRotSum = 0

    # frame: 0-299
    for frame in range(first_frame, last_frame):
        # 1-based index
        k = frame-first_frame+1

        # read current images
        I = imread(os.path.join(img_dir, '%06d.png'%frame)) # input image of current frame
        assert(len(I.shape) == 2) # image should be grayscale

        # read current frame features
        feature = np.load(osp.join(feature_dir, '%06d_%s.npy' % (frame, suffix))) # current frame features
        feature = np.ascontiguousarray(feature)
        feature = feature.astype(np.float32)

        # compute egomotion using libviso2 https://github.com/jlowenz/pyviso2
        # if_replace default is false, see https://github.com/jlowenz/pyviso2/blob/master/src/viso_mono.h
        process_result = visoMono.process_frame_preFeat(I, feature, if_replace)

        Tr = visoMono.getMotion() # delta pose matrix, change of pose wrt previous frame
        Tr_np = np.zeros((4, 4))
        Tr.toNumpy(Tr_np) # so awkward...

        # accumulate egomotion, starting with second frame
        if k > 1:
            if process_result is False:
                # if small/no motions are observed, set if_replace=True, 
                if_replace = True
                Tr_total.append(Tr_total[-1])
                Tr_total_np.append(Tr_total_np[-1])
            else:
                if_replace = False
                Tr_total.append(Tr_total[-1] * viso2.Matrix_inv(Tr))
                Tr_total_np.append(Tr_total_np[-1] @ np.linalg.inv(Tr_np)) # should be the same
                print(Tr_total_np[-1])

        # output statistics
        num_matches = visoMono.getNumberOfMatches()
        num_inliers = visoMono.getNumberOfInliers()
        matches = visoMono.getMatches()
        matches_np = np.empty([4, matches.size()])

        # dump all matched featrues in np array
        for i,m in enumerate(matches):
            matches_np[:, i] = (m.u1p, m.v1p, m.u1c, m.v1c)

        if if_vis:
            # update image
            ax1.clear()
            ax1.imshow(I, cmap='gray', vmin=0, vmax=255)
            if num_matches != 0:
                for n in range(num_matches):
                    ax1.plot([matches_np[0, n], matches_np[2, n]], [matches_np[1, n], matches_np[3, n]])
            ax1.set_title('Frame %d'%frame)

            # update trajectory
            if k > 1:
                ax2.plot([Tr_total_np[k-2][0, 3], Tr_total_np[k-1][0, 3]], \
                    [Tr_total_np[k-2][2, 3], Tr_total_np[k-1][2, 3]], 'b.-', linewidth=1)
                ax2.plot([gtTr[k-2][0, 3], gtTr[k-1][0, 3]], \
                    [gtTr[k-2][2, 3], gtTr[k-1][2, 3]], 'r.-', linewidth=1)
            ax2.set_title('Blue: estimated trajectory; Red: ground truth trejectory')

            plt.draw()

        # Compute rotation
        Rpred_p = Tr_total_np[k-2][0:3, 0:3]
        Rpred_c = Tr_total_np[k-1][0:3, 0:3]
        Rpred = Rpred_c.transpose() @ Rpred_p   # Rpred.shape = (3,3)
        Rgt_p = np.squeeze(gtTr[k-2, 0:3, 0:3])
        Rgt_c = np.squeeze(gtTr[k-1, 0:3, 0:3])
        Rgt = Rgt_c.transpose() @ Rgt_p         # Rgt.shape = (3,3)

        # Compute translation
        Tpred_p = Tr_total_np[k-2][0:3, 3:4]
        Tpred_c = Tr_total_np[k-1][0:3, 3:4]
        Tpred = Tpred_c - Tpred_p               # Tpred.shape = (3,1)
        Tgt_p = gtTr[k-2, 0:3, 3:4]
        Tgt_c = gtTr[k-1, 0:3, 3:4]
        Tgt = Tgt_c - Tgt_p                     # Tgt.shape = (3,1)

        # Compute errors
        errorRot, errorTrans = errorMetric(Rpred, Rgt, Tpred, Tgt)
        if frame == 20:
            print(f"Rpred type is {type(Rpred)}, shape is {Rpred.shape}")
            print(f"Rgt type is {type(Rgt)}, length is {Rgt.shape}")
            print(f"Tpred type is {type(Tpred)}, length is {Tpred.shape}")
            print(f"Tgt type is {type(Tgt)}, length is {Tgt.shape}")
            sys.exit()

        errorRotSum = errorRotSum + errorRot
        errorTransSum = errorTransSum + errorTrans
        print('Mean Error Rotation: %.5f'%(errorRotSum / (k-1)))
        print('Mean Error Translation: %.5f'%(errorTransSum / (k-1)))

        print('== [Result] Frame: %d, Matches %d, Inliers: %.2f'%(frame, num_matches, 100*num_inliers/(num_matches+1e-8)))

        if if_vis:
            # input('Paused; Press Enter to continue') # Option 1: Manually pause and resume
            if if_on_screen:
                plt.pause(0.1) # Or Option 2: enable to this to auto pause for a while after daring to enable animation in case of a delay in drawing
            vis_path = os.path.join(save_path, 'frame%03d.jpg'%frame)
            fig.savefig(vis_path)
            print('Saved at %s'%vis_path)

            if frame % 50 == 0 or frame == last_frame - 1:
                plt.figure(figsize=(10, 15))
                plt.imshow(plt.imread(vis_path))
                plt.axis('off')
                plt.show()


if __name__ == "__main__":
    dataset_path = '/datasets/cs252-sp21-A00-public/dataset_SfM'
    feature_dir = 'SIFT'
    runSFM(dataset_path, feature_dir)