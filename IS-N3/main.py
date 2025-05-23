import random
import cv2
import pytesseract
import os
import numpy as np
from imutils.perspective import four_point_transform

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_and_correct_table(image):
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Zgladi robove in zaznaj konture
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Poišči konture
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Poišči največji štirikotni obris
    table_contour = None
    max_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > max_area:
            table_contour = approx
            max_area = area

    if table_contour is None:
        raise Exception("Tabela ni bila najdena.")

    warped = four_point_transform(orig, table_contour.reshape(4, 2))
    return warped


def split_table_into_cells(image, rows, cols):
    h, w = image.shape[:2]
    cell_height = (h // rows)
    cell_width = (w // cols)
    cells = []

    for i in range(rows):
        for j in range(cols):
            x = j * cell_width
            y = i * cell_height
            cell = image[y:y + cell_height, x:x + cell_width]
            cells.append(((i, j), cell))  # Save position + cell
    return cells


def cluster_positions(pos, tol=5):
    # cluster positions that are close to each other
    if len(pos) == 0:
        return []
    groups = [[pos[0]]]
    for p in pos[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.mean(g)) for g in groups]


def get_grid_intersections(image, thresh_hold=230):
    img = image.copy()
    h, w = image.shape[:2]

    # convert to grayscale and binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh_hold, 255, cv2.THRESH_BINARY_INV)

    # get horizontal and vertical structuring elements - kernels
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))

    # erosion + dilation to remove noise and get long lines
    horiz_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_k, iterations=2)
    vert_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_k, iterations=2)

    # find intersections → grid corners
    inters = cv2.bitwise_and(horiz_lines, vert_lines)
    ys, xs = np.where(inters > 0)
    xs = np.unique(xs)
    ys = np.unique(ys)

    # cluster fount grid lines
    xs = cluster_positions(xs, tol=10)
    ys = cluster_positions(ys, tol=10)

    return xs, ys


def get_cell_from_image(image, xs, ys, padding=1):
    # get the cell from the image
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1 = xs[i] + padding
            y1 = ys[j] + padding
            x2 = xs[i + 1] - padding
            y2 = ys[j + 1] - padding
            cell = image[y1:y2, x1:x2]
            cells.append(((x1, y1), (x2, y2), cell, (i, j)))
    return cells


def debug_show_table(image, scale):
    h, w = image.shape[:2]
    resized_table = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow("Table", resized_table)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_show_intercections(image, xs, ys, scale):
    h, w = image.shape[:2]
    debug = image.copy()
    for y in ys:
        cv2.line(debug, (0, y), (w, y), (255, 0, 0), 1)
    for x in xs:
        cv2.line(debug, (x, 0), (x, h), (0, 255, 0), 1)
    for x in xs:
        for y in ys:
            cv2.circle(debug, (x, y), 5, (0, 0, 255), -1)

    debug = cv2.resize(debug, (int(w * scale), int(h * scale)))
    cv2.imshow("debug", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("number of intersections: ", len(xs) * len(ys))


def debug_show_cells(image, cells, scale):
    h, w = image.shape[:2]

    count = 0
    for cell in cells:
        (x1, y1), (x2, y2), cell_img, (i, j) = cell
        color = (0, 255, 0) if count % 3 == 0 else (255, 0, 0) if count % 3 == 1 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        count += 1

    resized_table = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow("Table", resized_table)
    cv2.imshow("cell1", random.choice(cells)[2])
    cv2.imshow("cell2", random.choice(cells)[2])
    cv2.imshow("cell3", random.choice(cells)[2])
    # cv2.imshow("cell", cells[50][2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def divide_into_quadrants(image, width_div, height_div):
    h, w = image.shape[:2]
    quad_h = h // height_div
    quad_w = w // width_div

    quadrants = []

    for i in range(height_div):
        for j in range(width_div):
            y1 = i * quad_h
            x1 = j * quad_w
            y2 = min((i + 1) * quad_h, h)
            x2 = min((j + 1) * quad_w, w)

            quadrant = image[y1:y2, x1:x2]
            quadrants.append((i, j, quadrant))

    return quadrants


def calculate_quadrant_features(quadrant):
    # Convert to grayscale if needed
    if len(quadrant.shape) > 2:
        gray = cv2.cvtColor(quadrant, cv2.COLOR_BGR2GRAY)
    else:
        gray = quadrant

    # Threshold to separate dark and light pixels
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    # Calculate features
    total_pixels = quadrant.size // (3 if len(quadrant.shape) > 2 else 1)
    dark_pixels = total_pixels - np.count_nonzero(binary)
    dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

    return dark_ratio


def process_table_image(image_path):
    image = cv2.imread(image_path)

    # show the table
    screen_width = 1920
    screen_height = 1080
    h, w = image.shape[:2]
    scale = min(screen_width / w, screen_height / h)

    table = detect_and_correct_table(image)
    # debug_show_table(table, scale)

    xs, ys = get_grid_intersections(table)
    # debug_show_intercections(table, xs, ys, scale)

    cells = get_cell_from_image(table, xs, ys, padding=2)
    debug_show_cells(table, cells, scale)


def process_all_fonts(input_base_dir, output_base_dir, width_div=4, height_div=4, file_count=20):
    # Create the main output directory
    os.makedirs(output_base_dir, exist_ok=True)
    base_features_file = os.path.join(output_base_dir, "features.csv")

    # Process each font subfolder
    for font_dir in ["vlke_tiskane", "vlke_pisane", "male_tiskane", "male_pisane"]:
        input_font_dir = os.path.join(input_base_dir, font_dir)
        output_font_dir = os.path.join(output_base_dir, font_dir)

        # Skip if directory doesn't exist
        if not os.path.exists(input_font_dir):
            print(f"Directory {input_font_dir} not found, skipping...")
            continue

        # Create output font directory and features directory
        os.makedirs(output_font_dir, exist_ok=True)

        # Prepare features CSV file
        features_file = os.path.join(output_font_dir, "features.csv")
        if not os.path.exists(features_file):
            with open(features_file, 'w', encoding='utf-8') as f:
                # Create header: letter + quadrant features
                header = ["letter_type"]

                # Add dynamic quadrant headers
                for i in range(height_div):
                    for j in range(width_div):
                        header.append(f"quadrant_{i}_{j}")

                f.write(",".join(header) + "\n")

        # Process each image in the font directory
        for filename in os.listdir(input_font_dir)[:file_count]:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_font_dir, filename)
                image_name = os.path.splitext(filename)[0]

                # Create output directory for this specific image
                output_image_dir = os.path.join(output_font_dir, image_name)
                os.makedirs(output_image_dir, exist_ok=True)

                print(f"Processing {image_path}...")
                try:
                    process_single_image(image_path, output_image_dir, font_dir, features_file, width_div, height_div)
                    # copy all lines from features_file to the base_features_file except first
                    with open(features_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[1:]
                    with open(base_features_file, 'a', encoding='utf-8') as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")


def process_single_image(image_path, output_dir, font_dir, features_file, width_div=4, height_div=4):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not load image {image_path}")

    # Detect and correct table
    table = detect_and_correct_table(image)

    # Get grid intersections
    xs, ys = get_grid_intersections(table)

    # Extract cells
    cells = get_cell_from_image(table, xs, ys, padding=2)

    if len(cells) != 50 * 25:
        raise Exception(f"Expected {50 * 25} cells, but found {len(cells)}")

    # get last cell
    (x1, y1), (x2, y2), cell_img, (i, j) = cells[-1]
    if i != 24 or j != 49:
        raise Exception(f"Last cell is not at the expected position (24, 49), but at ({i}, {j})")

    extract_features_from_cells(cells, output_dir, font_dir, features_file, 10, 7, True)
    list = [2, 3, 5, 10, 25, 50]
    for i in list:
        extract_features_from_cells(cells, output_dir, font_dir, features_file, i, i, False)


def extract_features_from_cells(cells, output_dir, font_label, features_file, width_div, height_div, first):
    abeceda = ["A", "B", "C", "C^", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "S^",
               "T", "U", "V", "Z", "Z^"]

    with open(features_file, 'a') as f:
        for (x1, y1), (x2, y2), cell_img, (i, j) in cells:
            cell_filename = f"{font_label}_{abeceda[i]}_{j}.png"
            if first:
                cell_path = os.path.join(output_dir, cell_filename)
                cv2.imwrite(cell_path, cell_img)

            quadrants = divide_into_quadrants(cell_img, width_div, height_div)
            row_data = [f"{font_label}_{abeceda[i]}"]

            for _, _, quadrant in quadrants:
                feature = calculate_quadrant_features(quadrant)
                row_data.append(str(round(feature, 4)))

            f.write(",".join(row_data) + "\n")


def extract_features_from_saved_cells(saved_cells_base_dir, output_base_dir, width_div=4, height_div=4):
    output_dir = os.path.join(output_base_dir, f"features_{width_div}x{height_div}")

    os.makedirs(output_dir, exist_ok=True)

    base_features_file = os.path.join(output_dir, "base_features.csv")

    # Pripravi header
    header = ["letter_type"] + [f"quadrant_{i}_{j}" for i in range(height_div) for j in range(width_div)]

    # Ustvari base features file
    with open(base_features_file, 'w', encoding='utf-8') as f:
        f.write(",".join(header) + "\n")

    for font_dir in os.listdir(saved_cells_base_dir):
        font_path = os.path.join(saved_cells_base_dir, font_dir)
        if not os.path.isdir(font_path):
            continue

        # Ustvari CSV za posamezen font_dir
        font_features_file = os.path.join(output_dir, f"{font_dir}_features.csv")
        with open(font_features_file, 'w', encoding='utf-8') as f:
            f.write(",".join(header) + "\n")

        for image_folder in os.listdir(font_path):
            image_folder_path = os.path.join(font_path, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            print(f"Processing: {image_folder_path}")

            for cell_file in sorted(os.listdir(image_folder_path)):
                if not cell_file.endswith(".png"):
                    continue

                cell_path = os.path.join(image_folder_path, cell_file)
                cell_img = cv2.imread(cell_path)
                if cell_img is None:
                    print(f"Could not read: {cell_path}")
                    continue

                quadrants = divide_into_quadrants(cell_img, width_div, height_div)
                row_data = ["_".join(cell_file.split(".")[0].split("_")[:-1])]

                for _, _, quadrant in quadrants:
                    feature = calculate_quadrant_features(quadrant)
                    row_data.append(str(round(feature, 4)))

                # Dodaj vrstico v oba CSV-ja
                line = ",".join(row_data) + "\n"
                with open(base_features_file, 'a', encoding='utf-8') as f:
                    f.write(line)
                with open(font_features_file, 'a', encoding='utf-8') as f:
                    f.write(line)


def find_and_copy_missing_files(source_dir, target_dir, output_dir):
    import os
    import shutil

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Collect all filenames from source_dir (abeceda) - just the base filenames
    source_filenames = set()
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_filenames.add(file)

    # Check target_dir (abeceda_testing) for files with names not in source
    missing_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            # Check if this filename is missing in source_dir
            if file not in source_filenames:
                src_path = os.path.join(root, file)
                missing_files.append((src_path, file))

    # Copy missing files to output_dir (flat structure)
    for src_path, filename in missing_files:
        # Create destination path
        dest_path = os.path.join(output_dir, filename)

        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {filename}")

    print(f"Total files copied: {len(missing_files)}")
    return missing_files


def test():
    input_base_dir = "abeceda"

    for font_dir in ["vlke_tiskane", "vlke_pisane", "male_tiskane", "male_pisane"]:
        input_font_dir = os.path.join(input_base_dir, font_dir)

        # Skip if directory doesn't exist
        if not os.path.exists(input_font_dir):
            print(f"Directory {input_font_dir} not found, skipping...")
            continue

        # Process each image in the font directory
        for filename in os.listdir(input_font_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_font_dir, filename)
                process_table_image(image_path)


if __name__ == "__main__":
    # test()

    # find_and_copy_missing_files("abeceda", "abeceda_testing", "missing_files")

    # process_all_fonts("abeceda", "output_abeceda", width_div=4, height_div=4, file_count=-1)

    extract_features_from_saved_cells("output_abeceda", "output_features", width_div=4, height_div=4)

    # list = [2,3,5,10,25,50]
    # for i in list:
    #     print(f"width_div: {i}, height_div: {i}")
    #     extract_features_from_saved_cells("output_abeceda", "output_features", width_div=i, height_div=i)
    #
    # extract_features_from_saved_cells("output_abeceda", "output_features", width_div=10, height_div=7)
