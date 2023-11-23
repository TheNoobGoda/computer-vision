from img_proc import ImgProc

def main():
    # hull = ImgProc.convexHull('img/pressedkeyboard.png')
    # ImgProc.crop_img('img/pressedkeyboard.png',hull)
    ImgProc.find_piano('img/keyboard_exemples/kbd1.jpeg')
    ImgProc.img_thresh('img/cropped_keyboard.jpg')


if __name__ == "__main__":
    main()