import numpy as np
import cv2

blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
white = (255, 255, 255)

def show_img(name, img):
  cv2.imshow(name, img)
  cv2.waitKey(0)

def plot_point(img, pt, color):
  cv2.circle(img, [int(round(v)) for v in pt], radius=3, color=color, thickness=-1)

def plot_rect(img, rect, w, h, color):
  cv2.rectangle(img,
                (int(round(rect[0] - w / 2)),
                 int(round(rect[1] - h / 2))),
                (int(round(rect[0] + w / 2)),
                 int(round(rect[1] + h / 2))),
                color=color, thickness=2)

def mask_grabcutbase(tp, w, h, img):
  rect = np.array((tp[0] - w // 2, tp[1] - h // 2, w, h)).astype(int)
  mask = np.zeros(img.shape[:2], np.uint8)
  cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
  mask = mask[:, :, np.newaxis]
  return mask

def mask_grabcut1(tp, w, h, img):
  mask = mask_grabcutbase(tp, w, h, img)
  return (mask == 2) | (mask == 0)
def mask_grabcut2(tp, w, h, img):
  return mask_grabcutbase(tp, w, h, img) == 0

def get_mslice(tp, w, h):
  return np.s_[tp[1] - h // 2 : tp[1] + round(h / 2),
               tp[0] - w // 2 : tp[0] + round(w / 2),
               :]

def mask_rect(tp, w, h, img):
  mask = np.ones_like(img)
  mask[get_mslice(tp, w, h)] = 0
  return mask

def mask_kmeans(tp, w, h, img):
  mslice = get_mslice(tp, w, h)
  pixs = img[mslice].astype(np.float32)
  pixs = pixs.reshape((w * h, 3))

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  k = 2
  _, labels, centers = cv2.kmeans(pixs, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  lab_mat = labels.reshape((h, w))
  #cent_s = lab_mat[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
  #center_ones = np.sum(cent_s)
  #if center_ones > 0.5 * w * h / 4:
  if np.linalg.norm(centers[1]) < np.linalg.norm(centers[0]):
    mask = np.where(lab_mat, 0, 1)
  else:
    mask = np.where(lab_mat, 1, 0)

  big_mask = np.ones_like(img)
  big_mask[mslice] = mask[:, :, np.newaxis]

  im = np.where(big_mask[mslice], 0, img[mslice])
  cv2.imshow('kmeans', im)

  return big_mask

def track_point(img1, img2, p):
  # Why are you the way that you are?
  prev_points = np.array(p, dtype=np.float32)[np.newaxis, np.newaxis, :]

  #win_size = (np.array((21, 21)) + 0.5 * dp[0][0]).astype(int)
  #win_size = np.array((21, 21)) + 2 * np.abs(dp[0][0].astype(int))
  win_size = np.array((21, 21))
  #new_points, status, err = cv2.calcOpticalFlowPyrLK(cropped, next_cropped, prev_points, prev_points + dp, winSize=win_size, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
  #new_points, status, err = cv2.calcOpticalFlowPyrLK(cropped, next_cropped, prev_points, prev_points, winSize=win_size, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
  new_points, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prev_points, None, winSize=win_size)
  return new_points[0][0]

class AutoOverlay:
  def __init__(self):
    self.tp = None

  def point_click(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.tp = np.array((2 * x, 2 * y))
      print("Got coords", *self.tp)

  def auto_overlay(self, *, videopath, start_time, crop_x, crop_y, crop_width, crop_height, rect_width, rect_height, frame_period, show_rect):
    vidcap = cv2.VideoCapture(videopath)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    work_window_name = 'CLICK ON OBJECT'

    tp_savefile = 'tps.txt'
    print(f"Saving tracked points to {tp_savefile}")

    # x, y, w, h
    crop_rect = (crop_x, crop_y, crop_width, crop_height)

    def crop(img):
      return img[crop_rect[1] : crop_rect[1] + crop_rect[3], crop_rect[0] : crop_rect[0] + crop_rect[2]]

    success, img = vidcap.read()
    h, w = img.shape[0], img.shape[1]
    cropped = crop(img)
    overlay = cropped.copy()

    #show_img(work_window_name, cropped)
    #rect = cv2.selectROI('select rectangle', cropped)

    #rect_buf = 5
    #cv2.rectangle(cropped, (rect[0] - rect_buf, rect[1] - rect_buf), (rect[0] + rect[2] + rect_buf, rect[1] + rect[3] + rect_buf), color=red, thickness=3)
    #show_img(work_window_name, cropped)

    #mask = np.zeros(cropped.shape[:2], np.uint8)
    #cv2.grabCut(cropped, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    #mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #masked = cropped * mask2[:, :, np.newaxis]
    #show_img('target', masked)

    cv2.namedWindow(work_window_name)
    cv2.setMouseCallback(work_window_name, self.point_click)

    # Tracked Point
    old_tp = None

    # Direction of travel
    #dp = np.zeros((1, 1, 2), dtype=np.float32)

    frames = []
    masks_used = []

    if True:
      tps = list(np.loadtxt(tp_savefile))
      print(f"Loaded {len(tps)} tracked points")

    i = 0
    frameind = 0
    while 1:
      newframe = False

      if frameind >= len(frames):
        assert frameind == len(frames)
        if self.tp is not None:
          cropped = frames[frameind - 1]

        for i in range(frame_period):
          success, next_img = vidcap.read()
          if self.tp is not None:
            next_cropped = crop(next_img)
            self.tp = track_point(cropped, next_cropped, self.tp)
            cropped = next_cropped
        frames.append(crop(next_img))

        newframe = True

      next_cropped = frames[frameind]

      if not newframe and frameind < len(tps):
        self.tp = tps[frameind]

      print("Ready. key for auto, 'x' for kmeans, 'z' for rect, enter for GrabCut, 'c' to skip frame, ESC to quit.")
      dummy = next_cropped.copy()
      if self.tp is not None:
        plot_point(dummy, self.tp, color=red)
        if show_rect:
          plot_rect(dummy, self.tp, rect_width, rect_height, color=red)

      showimg = cv2.resize(dummy, (dummy.shape[1] // 2, dummy.shape[0] // 2))
      cv2.imshow(work_window_name, showimg)
      key = cv2.waitKey(0)

      if key == 27:
        print("Got ESC, exiting.")
        break

      if self.tp is None:
        print("Must click initially!")
        continue

      #if old_tp is not None:
      #  dp[0][0] = (self.tp - old_tp) / frame_period

      if frameind >= len(tps):
        tps.append(self.tp)
      else:
        tps[frameind] = self.tp

      rounded_tp = [int(round(v)) for v in self.tp]

      maskf_args = (rounded_tp, rect_width, rect_height, next_cropped)

      if key == ord('h'): # Go back
        if frameind > 0:
          frameind -= 1
        else:
          print("AT START")
        continue

      elif key == ord('l'): # Go forwards
        if not newframe and frameind < len(frames) - 1:
          frameind += 1
        else:
          print("AT END")
        continue

      elif key == ord('a'):
        masks = []
        masks.append(mask_rect(*maskf_args))
        masks.append(mask_kmeans(*maskf_args))
        #gb_mask = mask_grabcutbase(*maskf_args)
        #masks.append((gb_mask == 2) | (gb_mask == 0))
        #masks.append(gb_mask == 0)

        names = ("rect", "prob background", "no prob bg")

        for j, mask in enumerate(masks):
          mslice = get_mslice(rounded_tp, rect_width, rect_height)
          im = np.where(mask[mslice], 0, next_cropped[mslice])
          cv2.imshow(names[j], im)

      elif key == 13: # Enter
        print("Running segmentation...")
        mask = mask_grabcut1(*maskf_args)
      elif key == ord('z'):
        print("Using rectangle block")
        mask = mask_rect(*maskf_args)
      elif key == ord('x'):
        print("Using kmeans")
        mask = mask_kmeans(*maskf_args)

      elif key == ord('c'):
        print("Skipping frame")
        mask = None
      elif key == ord(' '):
        if old_tp is None:
          print("Auto: rect (first)")
          mask = mask_rect(*maskf_args)
        else:
          ptd = np.abs(self.tp - old_tp)
          # TODO Use rect height too
          if np.all(ptd < rect_width * 0.9):
            print("Auto: kmeans (too close)")
            #mask = mask_grabcut1(*maskf_args)
            mask = mask_kmeans(*maskf_args)
          else:
            print("Auto: rect")
            mask = mask_rect(*maskf_args)
      else:
        print("Invalid key.")
        continue
      #hsv = cv2.cvtColor(next_cropped, cv2.COLOR_BGR2HSV)
      #hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)
      #brighter = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      #overlay = np.where(mask, overlay, brighter)

      old_tp = self.tp

      if frameind >= len(masks_used):
        masks_used.append(mask)
      else:
        masks_used[frameind] = mask

      #if mask is not None:
      #  overlay = np.where(mask, overlay, next_cropped)
      #  cv2.imwrite("overlay.png", overlay)
      #  cv2.imshow(work_window_name, overlay)
      #  cv2.waitKey(1)

      overlay = frames[0].copy()
      for j in range(frameind + 1):
        if masks_used[j] is not None:
          overlay = np.where(masks_used[j], overlay, frames[j])

      cv2.imwrite("overlay.png", overlay)
      showimg = cv2.resize(overlay, (overlay.shape[1] // 2, overlay.shape[0] // 2))
      cv2.imshow('overlay', showimg)
      cv2.waitKey(1)
      np.savetxt(tp_savefile, np.array(tps))

      frameind += 1

    #dp = new_points - prev_points

    cv2.destroyAllWindows()
