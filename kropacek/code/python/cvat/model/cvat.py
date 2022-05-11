from datetime import datetime

import pytz


class Polygon:
    def __init__(self, label: str, points: list, z_order: int, occluded: int = 0, group_id: int = 1, source: str = 'manual'):
        self.label = label
        self.occluded = occluded
        self.source = source
        self.points = points
        self.z_order = z_order
        self.group_id = group_id


class Image:
    def __init__(self, id: int, name: str, width: int, height: int):
        self.id = id
        self.name = name
        self.width = width
        self.height = height
        self.polygons = list()

    def add_polygon(self, polygon: Polygon):
        self.polygons.append(polygon)


class Label:
    def __init__(self, name: str, cvat_color: str, label_img_color: tuple):
        self.name = name
        self.color = cvat_color
        self.label_img_color = label_img_color

    def __eq__(self, other):
        if not isinstance(other, Label):
            return False
        return self.name == other.name


class Segment:
    def __init__(self, id: int, start: int, stop: int):
        self.id = id
        self.start = start
        self.stop = stop


class Task:
    def __init__(self, id: int, name: str, size: int, labels: list, segments: list,
                 z_order: bool = True, mode: str = 'annotation', overlap: int = 0):
        self.id = id
        self.name = name
        self.size = size
        self.mode = mode
        self.overlap = overlap
        self.z_order = z_order
        self.labels = labels
        self.segments = segments
        self.start_frame = 0
        self.stop_frame = self.size - 1
        self.created = datetime.now(pytz.utc).isoformat()
        self.updated = datetime.now(pytz.utc).isoformat()

    def add_label(self, label: Label):
        self.labels.append(label)

    def add_segment(self, segment: Segment):
        self.segments.append(segment)


class Meta:
    def __init__(self, task: Task):
        self.task = task


class Annotations:
    def __init__(self, meta: Meta, images: list, version: str = '1.1'):
        self.version = version
        self.meta = meta
        self.images = images

    def add_image(self, image: Image):
        self.images.append(image)
