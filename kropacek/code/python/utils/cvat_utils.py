def get_categories_to_colors():
    return {
        'Background': (0, 0, 0),
        'Vessel': (0, 0, 255),
        'Lumen': (255, 0, 0),
        'Plaque': (0, 255, 0),
    }


def points_to_string(xy):
    result = []
    for i in xy:
        result.append(','.join([str(i[0]), str(i[1])]))
    return ';'.join(result)
