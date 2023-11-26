from img_proc import ImgProc

def main():
    # hull = ImgProc.convexHull('img/pressedkeyboard.png')
    # ImgProc.crop_img('img/pressedkeyboard.png',hull)
    ImgProc.find_piano('img/keyboard_exemples/kbd1.jpeg')
    black_kesy, white_keys = ImgProc.find_keys('img/results/cropped_keyboard.jpg')
    print(len(black_kesy),len(white_keys))


if __name__ == "__main__":
    main()