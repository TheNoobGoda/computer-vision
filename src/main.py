from img_proc import ImgProc

def main():
    hull = ImgProc.convexHull('img/keyboard_unpressed.jpg')
    ImgProc.edge_lines('img/keyboard_unpressed.jpg')
    ImgProc.crop_img('img/keyboard_unpressed.jpg',hull)
    ImgProc.img_thresh('img/cropped_keyboard.jpg')


if __name__ == "__main__":
    main()