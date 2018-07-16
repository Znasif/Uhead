from Visual import Visual
from Process import Process


def sp(t):
    return t.split('.')[0]


def make_subplots(t):
    """
    Stores a sub-categorized image in the title+"sub.jpg" file
    :param t: file name
    :return: NIL
    """
    a = Visual.image_open(t)
    b = Visual.image_open(t, 1)
    a, b = Process.get_contour(a, 1, b)
    # Visual.image_write(sp(t) + "_subs", a)
    Visual.plot(sp(t) + "_subs", a)


def separate_plots(t):
    """
    Separate the image into two different images
    :param t: file name
    :return: NIL
    """
    a = Visual.image_open(t)
    a, b = Process.get_split(a)
    # Visual.image_write(sp(t) + "_split_to_plot", a)
    # Visual.image_write(sp(t) + "_split_to_num", b)
    Visual.plot(sp(t) + "_split_to_plot", a)
    Visual.plot(sp(t) + "_split_to_num", b)
    c = Visual.get_overlay(a, b)
    # Visual.image_write(sp(t) + "_overlay", c)
    Visual.plot(sp(t) + "_overlay", c)


def hough_trans(t):
    """
    Apply probabilistic Hough Transform to determine possible missing edges
    :param t: file name
    :return: NIL
    """
    a = Visual.image_open(t)
    a, b = Process.get_split(a)
    a = Process.find_hough(a)
    # Visual.image_write(sp(t) + "_houghl", a)
    Visual.plot(sp(t) + "_houghl", a)


def manual(t):
    """
    Image generated from manual edge linking
    :param t: file name
    :return: NIL
    """
    a = Visual.image_open(t)
    b = Visual.image_open(t, 1)
    Visual.get_pixel(a)
    a, c = Process.get_contour(Visual.image, 1, b)
    Visual.image_write(sp(t) + "_manual", a)


if __name__ == "__main__":
    title = ["Numbered.png", "Enhanced.png", "nums.jpg", "plot.jpg", "port.jpg", "see.jpg", "subsection.jpg",
             "testplot.jpg", "trial.jpg"]
    temps = ["Template/Symbols.png", "Template/SymbolSet.png", "Template/Crops"]

    # make_subplots(title[0])
    # separate_plots(title[4])
    # hough_trans(title[3])
    manual(title[3])
