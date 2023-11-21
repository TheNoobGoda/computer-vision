from img_proc import ImgProc

def main():
    hull = ImgProc.convexHull('img/pressedkeyboard.png')
    ImgProc.get_keyboard('img/pressedkeyboard.png',hull)


if __name__ == "__main__":
    main()