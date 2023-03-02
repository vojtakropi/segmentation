from lxml import etree

from export.cvat.model.cvat import Annotations, Meta, Task
from utils.cvat_utils import points_to_string


def create_sub_element_value(element, name:str, value):
    tag = etree.SubElement(element, name)
    tag.text = str(value)
    return tag


def create_attribute_value(element, name: str, obj):
    element.set(name, str(obj))


def create_attribute_value_dynamic(element, name: str, obj):
    create_attribute_value(element, name, getattr(obj, name))


def create_sub_element_value_dynamic(element, name: str, obj):
    return create_sub_element_value(element, name, getattr(obj, name))


def map_labels(element, labels: list):
    if labels is None:
        return
    labels_tag = etree.SubElement(element, 'labels')
    for label in labels:
        label_tag = etree.SubElement(labels_tag, 'label')
        create_sub_element_value_dynamic(label_tag, 'name', label)
        create_sub_element_value_dynamic(label_tag, 'color', label)
    return labels_tag


def map_segments(element, segments: list):
    if segments is None:
        return
    segments_tag = etree.SubElement(element, 'segments')
    for segment in segments:
        label_tag = etree.SubElement(segments_tag, 'segment')
        create_sub_element_value(label_tag, 'id', segment)
        create_sub_element_value(label_tag, 'start', segment)
        create_sub_element_value(label_tag, 'stop', segment)
        create_sub_element_value(label_tag, 'url', segment)
    return segments_tag


def map_task(element, task: Task):
    if task is None:
        return
    task_tag = etree.SubElement(element, 'task')
    sub_element_names = ['id', 'name', 'size', 'mode', 'overlap', 'created', 'updated', 'start_frame', 
                         'stop_frame', 'z_order']
    for name in sub_element_names:
        create_sub_element_value_dynamic(task_tag, name, task)
    map_labels(task_tag, task.labels)
    map_segments(task_tag, task.segments)
    return task_tag
    

def map_meta(element, meta: Meta):
    if meta is None:
        return
    meta_tag = etree.SubElement(element, 'meta')
    map_task(meta_tag, meta.task)
    return meta_tag


def map_polygons(element, polygons: list):
    if polygons is None:
        return
    for polygon in polygons:
        polygon_tag = etree.SubElement(element, 'polygon')
        create_attribute_value_dynamic(polygon_tag, "label", polygon)
        create_attribute_value_dynamic(polygon_tag, "occluded", polygon)
        create_attribute_value_dynamic(polygon_tag, "source", polygon)
        create_attribute_value(polygon_tag, "points", points_to_string(polygon.points))
        create_attribute_value_dynamic(polygon_tag, "z_order", polygon)
        create_attribute_value_dynamic(polygon_tag, "group_id", polygon)
    return element


def map_images(element, images: list):
    if images is None:
        return
    for image in images:
        image_tag = etree.SubElement(element, 'image')
        create_attribute_value_dynamic(image_tag, "id", image)
        create_attribute_value_dynamic(image_tag, "name", image)
        create_attribute_value_dynamic(image_tag, "width", image)
        create_attribute_value_dynamic(image_tag, "height", image)
        map_polygons(image_tag, image.polygons)
    return element


def map_to_xml(annotations: Annotations):
    root_tag = etree.Element('annotations')

    version_tag = etree.SubElement(root_tag, 'version')
    version_tag.text = annotations.version

    map_meta(root_tag, annotations.meta)

    map_images(root_tag, annotations.images)

    return root_tag


