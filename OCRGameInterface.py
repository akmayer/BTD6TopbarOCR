from . import OCR
import pyautogui, keyboard, time, cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import importlib, re, math, json
from win32gui import GetWindowText, GetForegroundWindow
import os
from PIL import ImageDraw, ImageFont
from pathlib import Path
from scipy.signal import convolve2d, correlate2d
from scipy.ndimage import maximum_filter



BASE_DIR = Path(__file__).resolve().parent


def text_with_default_stroke(text, font_path, font_size=64, padding=20, stroke_width=6):
    font = ImageFont.truetype(font_path, font_size)

    # Measure text
    dummy = Image.new("L", (1, 1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Create image
    img = Image.new("L", (w + 2*padding, h + 2*padding), 255)
    draw = ImageDraw.Draw(img)

    # Draw text with stroke
    draw.text(
        (padding, padding),
        text,
        font=font,
        fill=0,              # white fill
        stroke_width=stroke_width,
        stroke_fill=255          # black outline
    )

    return np.array(img)
    
def genImageTemplates(charSet = '1234567890'):
    templateSet = {}
    for char in charSet:
        arr = text_with_default_stroke(
            char,
            BASE_DIR / "LuckiestGuy-Regular.ttf",
            font_size=146,
            stroke_width = 12,
            padding=10
        )
        templateSet[char] = arr
    return templateSet

def normxcorr2(image, kernel):
    image = image.astype(float)
    kernel = kernel.astype(float)
    h, w = kernel.shape
    
    kernel_mean = np.mean(kernel)
    kernel = kernel - kernel_mean
    kernel_std = np.std(kernel)
    
    # numerator
    numerator = correlate2d(image, kernel, mode='same')
    
    # local mean of image
    ones = np.ones((h, w))
    local_sum = correlate2d(image, ones, mode='same')
    local_mean = local_sum / (h * w)
    
    # local std
    local_sq_sum = correlate2d(image**2, ones, mode='same')
    local_var = local_sq_sum / (h * w) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-8))
    
    return numerator / (local_std * kernel_std * h * w)

def nms_points(xs, ys, scores, radius=5):
    indices = np.argsort(scores)[::-1]  # sort by score descending
    
    keep = []
    taken = np.zeros(len(xs), dtype=bool)
    
    for i in indices:
        if taken[i]:
            continue
        
        keep.append(i)
        
        # suppress nearby points
        for j in range(len(xs)):
            if not taken[j]:
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < radius:
                    taken[j] = True
    
    return ys[keep], xs[keep]
def suppress_nearby(char_vals, dist_thresh=10):
    """
    char_vals: list of (char, x, confidence)
    returns: filtered list with nearby duplicates suppressed
    """
    # Sort by confidence descending
    sorted_vals = sorted(char_vals, key=lambda x: x[2], reverse=True)
    
    kept = []
    
    for char, x, conf in sorted_vals:
        keep = True
        
        for _, kept_x, _ in kept:
            if abs(x - kept_x) <= dist_thresh:
                keep = False
                break
        
        if keep:
            kept.append((char, x, conf))
    
    # Optional: sort back by x position (left → right)
    kept.sort(key=lambda x: x[1])
    
    return kept

def compress_whitespace_columns(img, max_gap=240, white_thresh=250):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    # Detect white columns
    # A column is "white" if all pixels are above threshold
    white_cols = np.all(img >= white_thresh, axis=0)

    result_columns = []
    i = 0

    while i < w:
        if white_cols[i]:
            # Start of a whitespace run
            start = i
            while i < w and white_cols[i]:
                i += 1
            end = i

            run_width = end - start

            if run_width <= max_gap:
                # Keep as is
                result_columns.append(img[:, start:end])
            else:
                # Compress to max_gap
                result_columns.append(img[:, start:start+max_gap])

        else:
            # Non-white column (text)
            start = i
            while i < w and not white_cols[i]:
                i += 1
            end = i

            result_columns.append(img[:, start:end])

    # Concatenate all pieces
    result = np.hstack(result_columns)
    return result

def downsample_image(image: np.ndarray, factor: int = 6) -> np.ndarray:
    """
    Downsample a NumPy image using PIL.

    Args:
        image (np.ndarray): Input image (H, W) or (H, W, C)
        factor (int): Downsampling factor (default = 8)

    Returns:
        np.ndarray: Downsampled image
    """
    if factor <= 0:
        raise ValueError("factor must be a positive integer")

    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(image)

    # Compute new size
    new_width = max(1, pil_img.width // factor)
    new_height = max(1, pil_img.height // factor)

    # Resize using high-quality downsampling filter
    pil_resized = pil_img.resize((new_width, new_height))

    # Convert back to NumPy array
    return np.array(pil_resized)


def roll_right_region_vertically(img, split_percent=0.5, shift_pixels=10):
    """
    Rolls the right portion of an image vertically.

    Parameters:
        img (np.ndarray): Input image (H x W) or (H x W x C)
        split_percent (float): Fraction across width where rolling starts (0–1)
        shift_pixels (int): Number of pixels to roll vertically

    Returns:
        np.ndarray: Modified image
    """
    h, w = img.shape[:2]
    
    # Compute split column
    split_col = int(w * split_percent)
    
    # Copy to avoid modifying original
    result = img.copy()
    
    # Roll the right region vertically (axis=0)
    result[:, split_col:] = np.roll(
        result[:, split_col:], 
        shift=shift_pixels, 
        axis=0
    )
    
    return result

def pad_top_white(img, pad_height):
    """
    Pads the top of an image with white pixels.

    Parameters:
        img (np.ndarray): Input image (H x W) or (H x W x C)
        pad_height (int): Number of pixels to add at the top

    Returns:
        np.ndarray: Padded image
    """
    if img.ndim == 2:  # grayscale
        return np.pad(img, ((pad_height, 0), (0, 0)), constant_values=255)
    else:  # color image
        return np.pad(img, ((pad_height, 0), (0, 0), (0, 0)), constant_values=255)




def extractIntegerGroups(inputString):
    # Use regular expression to find all groups of integers
    groups = re.findall(r'\d+', inputString)
    
    # Convert strings to integers and return
    return [int(group) for group in groups]

def flood_fill(image, startCoord = (0, 0)):
    # Convert image to uint8 for compatibility with OpenCV
    image_uint8 = image.astype(np.uint8)
    image_uint8[0, 0]=255

    # Perform flood fill
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(image_uint8, mask, (0, 0), 0, 200, 255, cv2.FLOODFILL_FIXED_RANGE)

    # Convert back to original dtype
    filled_image = image_uint8.astype(image.dtype)
    
    return filled_image

def process(data):
    data = sorted(data, key=lambda x: x[1])
    chars = [c for c, _, _ in data]
    xs = np.array([x for _, x, _ in data], dtype=float)
    n = len(xs)

    if n <= 1:
        return (int(chars[0]) if n == 1 else 0, 0, 0, 0)

    gaps = [(xs[i+1] - xs[i], i) for i in range(n-1)]

    # -------------------------
    # CASE 1: exactly 1 big gap → 2 groups
    # -------------------------
    big30 = [(g, i) for g, i in gaps if g >= 30]
    if len(big30) == 1:
        _, idx = big30[0]
        groups = [chars[:idx+1], chars[idx+1:]]

    # -------------------------
    # CASE 2: ≥ 2 big gaps → 4 groups
    # -------------------------
    else:
        big40 = [(g, i) for g, i in gaps if g >= 40]
        if len(big40) < 2:
            # problem constraint says this should not happen
            raise ValueError("Invalid input: expected at least 2 large gaps")

        # take the two largest big gaps
        big40_sorted = sorted(big40, reverse=True)
        primary_splits = sorted([i for g, i in big40_sorted[:2]])

        # build segments using ONLY these coarse splits
        segments = []
        start = 0
        for s in primary_splits:
            segments.append((start, s))
            start = s + 1
        segments.append((start, n - 1))

        # -------------------------
        # refine ONLY last segment (this is the key fix)
        # -------------------------
        l, r = segments[2]

        # find SMALL gap ≥ 12 inside last segment
        refinement = None
        for i in range(r-1, l-1, -1):   # scan from right
            if xs[i+1] - xs[i] >= 20:
                refinement = i
                break

        if refinement is None:
            raise ValueError("No valid refinement gap found in group 3-4 split")

        # final split indices
        split1, split2 = primary_splits
        split3 = refinement

        split_indices = sorted([split1, split2, split3])

        groups = []
        start = 0
        for s in split_indices:
            groups.append(chars[start:s+1])
            start = s+1
        groups.append(chars[start:])

    ints = [int(''.join(g)) for g in groups]
    while len(ints) < 4:
        ints.append(0)

    return tuple(ints[:4])

def matchImageToTemplateSet(inputImage, templateSet,
                            threshold = 0.7,
                            maxFilterSize = 8,
                            maxSuppressionSize = 8,
                           competingCharSuppressionSize = 8):

    characterColValues = []
    
    s = time.time()
    
    for character, template in templateSet.items():
        convInput = (inputImage[::6, ::6]).astype(float)
        convKernel = (template[::6, ::6]).astype(float)
        
        ncc = normxcorr2(convInput, convKernel)
        neighborhood = maximum_filter(ncc, size=maxFilterSize)
        peaks = (ncc == neighborhood) & (ncc > threshold)
        
        ys, xs = np.where(peaks)
        
        scores = ncc[ys, xs]
        ys_nms, xs_nms = nms_points(xs, ys, scores, radius=maxSuppressionSize)

        for rowValue, colValue in zip(ys_nms, xs_nms):
            characterColValues.append((character, colValue, ncc[rowValue, colValue]))
        characterColValues = sorted(characterColValues, key = lambda x : x[1])
        characterColValues = suppress_nearby(characterColValues, dist_thresh = competingCharSuppressionSize)
    topbarData = process(characterColValues)
    return topbarData


class GameInterface():
    def __init__(self):
        package_dir = os.path.dirname(__file__)

        self.templateSet = genImageTemplates()

        self.proccedImages = []
    
    def analyzeTopbar(self):
        self.topbar = pyautogui.screenshot(region=(130, 15, 1488-130+80, 70))        
        
        self.topbarProc = np.array(self.topbar)


        hardcodedMoneyIconTargetColor= np.array([255.0, 189.66666667, 0.0])

        diff = self.topbarProc - hardcodedMoneyIconTargetColor  # shape (h, w, 3)
        dist2 = np.sum(diff**2, axis=-1)  # shape (h, w)
        
        h_idx, w_idx = np.unravel_index(np.argmin(dist2), dist2.shape)

        # Gray out this annoying af $ icon
        self.topbarProc[:, w_idx + 30: w_idx + 50, :] = np.array([100., 100, 100])

                        
        #Convert whole image to grayscale
        self.topbarGray = cv2.cvtColor(self.topbarProc, cv2.COLOR_BGR2GRAY)
        
        #CRUX OF CONSISTENT PREPROCESSING FOR OCR: !!!!!
        #Masking white pixels to get the text alone is not enough since ice monkeys, some backgrounds, etc have VERY white pixels
        #Causes a significant amount of noise that frequently results in error
        #Needs to be used in conjunction with someway of seperating the background from the foreground, so that's what this does:
        
        #Gets a binary image where every light pixel gets set to a value of 255
        #Goal is to extract the black outlines around the numbers
        self.topbarBin = (self.topbarGray > 30) * 255
        
        #Makes the whole border white to make the background floodfill more consistent
        self.topbarBin[0,:] = 255
        self.topbarBin[-1,:] = 255
        self.topbarBin[:,0] = 255
        self.topbarBin[:,-1] = 255
        
        #Flood fills any bright pixels from the top left corner to be black.
        #This mask blacks out any brightish pixels but ONLY in the background since it stops at the black number outlines!
        self.numberMask = flood_fill(self.topbarBin)
        
        #Applies the mask to the original photo so that the only bright pixels left in the original photo are the number values of interest!
        self.topbarProc[~(self.numberMask != 0)] = 0

        min_vals = self.topbarProc.min(axis=2)  # shape (H, W)

        self.topbarProc = np.stack([min_vals]*3, axis=2)

        self.topbarProc = 255 - self.topbarProc

        self.topbarProc[self.topbarProc >= 5] = 255


        gray = cv2.cvtColor(self.topbarProc, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.topbarProc = compress_whitespace_columns(thresh)
        #self.topbarProc = downsample_image(self.topbarProc)
        self.proccedImages.append(self.topbarProc)
        #plt.figure(figsize=(12, 8), dpi=200)
        #plt.imshow(self.topbarProc, interpolation = 'none')
        #plt.show()
        self.topbarData = matchImageToTemplateSet(self.topbarProc, self.templateSet, threshold = 0.65)
        
        #print(self.topbarData)
        #self.topbarData = ' '.join(result.txts)
        #self.topbarData = re.sub(r'\.', '', self.topbarData)
        #self.topbarData = re.sub(r',', '', self.topbarData)
        #self.topbarData = re.sub(r'[^\d\s]', ' ', self.topbarData)
        #self.topbarData = extractIntegerGroups(self.topbarData)
        
    def getHealth(self):
        return self.topbarData[0]
        
    def getMoney(self):
        return self.topbarData[1]
    
    def getRound(self):
        return self.topbarData[2]
