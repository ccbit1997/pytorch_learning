from read_data import MyData
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    root_dir = "dataset/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)
    img, label = ants_dataset[0]
    img.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
