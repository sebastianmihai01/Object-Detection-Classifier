import os

# Compression for Google Colab
LABELIMG_PATH = os.path.join('../Tensorflow', 'labelimg')
# Compression for Google Colab
TRAIN_PATH = os.path.join('../Tensorflow', 'workspace', 'images', 'train')
# Compression for Google Colab
TEST_PATH = os.path.join('../Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('../Tensorflow', 'workspace', 'images', 'archive.tar.gz')
path = "/Tensorflow/labelimg"
cd2 = "cd C:\\Users\\Sebi\\Desktop\\AI Projects\\Object-Detection-Classifier\\Tensorflow"

cmd1 = "pip install --upgrade pyqt5 lxml"
cmd2 = "mkdir " + LABELIMG_PATH
cmd3 = cd2 + "&& git clone https://github.com/tzutalin/labelImg "
cmd4 = "make qt5py3"
cmd5 = "cd " + LABELIMG_PATH + " && pyrcc5 -o libs/resources.py resources.qrc"
cmd6 = "tar -czf " + ARCHIVE_PATH + " " + TRAIN_PATH + " " + TEST_PATH
cd = "cd C:\\Users\\Sebi\\Desktop\\AI Projects\\Object-Detection-Classifier"



def execute(cmd):
    os.system(cmd)

print(LABELIMG_PATH)


def __main__():
    if not os.path.exists(LABELIMG_PATH):
        print("1")
        execute(cd)
        print("2")
        execute(cmd2)
        print("3")
        execute(cd2)
        print("4")
        execute(cmd3)
        print("5")

    if os.name == 'posix':
        execute(cmd4)
    if os.name == 'nt':
        execute(cmd5)

    print("6")
    execute("cd " + LABELIMG_PATH + " && python labelImg.py")
    print("7")
    execute(cmd6)
    print("8")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Please choose: Open Dir -> Navigate to Classes Directory (with sample images) \n")
    print("Make sure the items you want to select appear on the lower-right corner of the application \n")
    print("Please fit the image that you want to label perfectly, don't let any foreign object within the frame\n")
    print("Hit CTRL+S to save \n")
    print("The new file will have the exact name as the image, but with a .xml extension, where "
          "the bdnbox shows the coordinates of the label")
    print("-----------------------------------------------------------------------------------------------------------")



__main__()
