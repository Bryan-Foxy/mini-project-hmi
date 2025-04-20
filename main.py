from front import MainWindow

if __name__ == "__main__":
    # Initialize the MainWindow with the path to the video
    path = "datas/IMG_5916.MOV"
    main_window = MainWindow(video_path = path)
    # Run
    main_window.run()