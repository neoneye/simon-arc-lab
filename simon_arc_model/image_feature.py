from enum import Enum

class ImageFeature(Enum):
    COMPONENT_NEAREST4 = 'component_nearest4'
    COMPONENT_ALL8 = 'component_all8'
    COMPONENT_CORNER4 = 'component_corner4'
    SUPPRESS_CENTER_PIXEL_ONCE = 'suppress_center_pixel_once'
    SUPPRESS_CENTER_PIXEL_LOOKAROUND = 'suppress_center_pixel_lookaround'
    HISTOGRAM_DIAGONAL = 'histogram_diagonal'
    HISTOGRAM_ROWCOL = 'histogram_rowcol'
    HISTOGRAM_VALUE = 'histogram_value'
    IMAGE_MASS_COMPARE_ADJACENT_ROWCOL = 'image_mass_compare_adjacent_rowcol'
    IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 = 'image_mass_compare_adjacent_rowcol2'
    NUMBER_OF_UNIQUE_COLORS_ALL9 = 'number_of_unique_colors_all9'
    NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER = 'number_of_unique_colors_around_center'
    NUMBER_OF_UNIQUE_COLORS_IN_CORNERS = 'number_of_unique_colors_in_corners'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 = 'number_of_unique_colors_in_diamond4'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 = 'number_of_unique_colors_in_diamond5'
    ROTATE45 = 'image_rotate45'
    COUNT_NEIGHBORS_WITH_SAME_COLOR = 'count_neighbors_with_same_color'
    EROSION_ALL8 = 'erosion_all8'
    EROSION_NEAREST4 = 'erosion_nearest4'
    EROSION_CORNER4 = 'erosion_corner4'
    EROSION_ROWCOL = 'erosion_rowcol'
    EROSION_DIAGONAL = 'erosion_diagonal'
    ANY_CORNER = 'any_corner'
    CORNER = 'corner'
    ANY_EDGE = 'any_edge'
    CENTER = 'center'
    BOUNDING_BOXES = 'bounding_boxes'
    DISTANCE_INSIDE_OBJECT = 'distance_inside_object'
    GRAVITY_DRAW_TOP_TO_BOTTOM = 'gravity_draw_top_to_bottom'
    GRAVITY_DRAW_BOTTOM_TO_TOP = 'gravity_draw_bottom_to_top'
    GRAVITY_DRAW_LEFT_TO_RIGHT = 'gravity_draw_left_to_right'
    GRAVITY_DRAW_RIGHT_TO_LEFT = 'gravity_draw_right_to_left'
    GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT = 'gravity_draw_topleft_to_bottomright'
    GRAVITY_DRAW_BOTTOMRIGHT_TO_TOPLEFT = 'gravity_draw_bottomright_to_topleft'
    GRAVITY_DRAW_TOPRIGHT_TO_BOTTOMLEFT = 'gravity_draw_topright_to_bottomleft'
    GRAVITY_DRAW_BOTTOMLEFT_TO_TOPRIGHT = 'gravity_draw_bottomleft_to_topright'
    POSITION_XY0 = 'position_xy0'
    POSITION_XY4 = 'position_xy4'
    OBJECT_ID_RAY_LIST = 'object_id_ray_list'
    IDENTIFY_OBJECT_SHAPE = 'identify_object_shape'
    BIGRAM_ROWCOL = 'bigram_rowcol'
    COLOR_POPULARITY = 'color_popularity'
    SHAPE_ALL8 = 'shape_all8'

    @classmethod
    def names_joined_with_comma(cls, features: set['ImageFeature']) -> str:
        """
        Human readable compact representation of multiple features.
        """
        names_unordered = [feature.name for feature in features]
        names_sorted = sorted(names_unordered)
        return ','.join(names_sorted)
