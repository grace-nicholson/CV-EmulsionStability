from dataclasses import dataclass, field

# Example calls to DataClasses:
# for idx, v in enumerate(detections_class.vials):
#     print('Vial ', idx)
#     print(v)

@dataclass
class Phase:
    phase_ind: int
    phase_name: str
    phase_conf: float
    phase_bbox_x1y1_x2y2: list[float]


@dataclass
class Vial:
    vial_ind: int

    # Formulation metadata
    sample_id: str
    surfactant: str
    surfactant_conc: float
    oil: str
    oil_conc: float
    pH: float
    time_s: float

    # YOLO vial detection
    vial_conf: float
    vial_bbox_x1y1_x2y2: list[float]

    # Detected phases
    phases: list[Phase] = field(default_factory=list)

    flag: str = ""


@dataclass
class ImageResult:
    filename: str
    vials: list[Vial] = field(default_factory=list)
