from ultralytics import YOLO
from data_models import Vial, Phase, ImageResult
import json
import csv
from dataclasses import asdict
import os
import glob
import pandas as pd

PHASE_CLASSES = {
    "oil",
    "surfactant",
    "separating",
    "emulsion",
    "creaming",
    "continuous",
    "sedimentation"
}

def box_center(box):
    """Return the centre x,y coordinate of a bounding box."""
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def point_inside_box(point, box):
    """Check whether a point is inside a bounding box."""
    x, y = point
    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2

def save_json(image_results, filename):

    with open(filename, "w") as f:
        json.dump(
            [asdict(image_result) for image_result in image_results],
            f,
            indent=4,
            default=str
        )

def save_csv(image_results, filename):

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "vial_ind",

            # Formulation metadata
            "sample_id",
            "surfactant",
            "surfactant_conc",
            "oil",
            "oil_conc",
            "pH",
            "time_s",

            # Vial detection
            "vial_conf",
            "vial_x1",
            "vial_y1",
            "vial_x2",
            "vial_y2",

            # Phase detection
            "phase_ind",
            "phase_name",
            "phase_conf",
            "phase_x1",
            "phase_y1",
            "phase_x2",
            "phase_y2",
        ])

        # Go through every image
        for image_result in image_results:

            # Go through every vial in that image
            for vial in image_result.vials:

                vx1, vy1, vx2, vy2 = vial.vial_bbox_x1y1_x2y2

                # Go through every phase in that vial
                for phase in vial.phases:

                    px1, py1, px2, py2 = phase.phase_bbox_x1y1_x2y2

                    writer.writerow([
                        image_result.filename,
                        vial.vial_ind,

                        # Formulation metadata
                        vial.sample_id,
                        vial.surfactant,
                        vial.surfactant_conc,
                        vial.oil,
                        vial.oil_conc,
                        vial.pH,
                        vial.time_s,

                        # Vial detection
                        vial.vial_conf,
                        vx1,
                        vy1,
                        vx2,
                        vy2,

                        # Phase detection
                        phase.phase_ind,
                        phase.phase_name,
                        phase.phase_conf,
                        px1,
                        py1,
                        px2,
                        py2
                    ])
def main():

    # Load model
    model = YOLO("yolo26m_pt_barkla/weights/best.pt")

    # Load formulation metadata
    metadata = pd.read_csv("LF_RS_pH5_data.csv")

    # Folder containing images
    image_folder = "/Users/grace/msc_project/Training_code/images"

    # Find all PNG images in folder
    image_files = sorted(
    glob.glob(os.path.join(image_folder, "*.png")),
    key=lambda x: int(
        os.path.basename(x)
        .replace("frame_", "")
        .replace("s.png", "")
    )
)

    print(f"Found {len(image_files)} images.")

    # Store results from ALL images
    all_image_results = []


    # Process every image
    for image_number, source in enumerate(image_files, start=0):

        print(
            f"\nProcessing image "
            f"{image_number}/{len(image_files)}: "
            f"{os.path.basename(source)}"
        )

        # Get image filename
        image_name = os.path.basename(source)

        # Run YOLO
        results = model(source)

        # Create ImageResult
        image_result = ImageResult(filename=source)


        for result in results:

            # Store detections temporarily
            vial_detections = []
            phase_detections = []

            for box in result.boxes:

                class_ind = int(box.cls[0])
                class_name = result.names[class_ind]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if class_name == "Vial":

                    vial_detections.append({
                    "class_ind": class_ind,
                    "confidence": confidence,
                    "bbox": bbox
                })

                elif class_name in PHASE_CLASSES:

                    phase_detections.append({
                        "class_ind": class_ind,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })

            # Sort vials from left to right
            vial_detections = sorted(
                vial_detections,
                key=lambda vial: vial["bbox"][0]
            )

            # Associate phases with vials
            for vial_index, vial_detection in enumerate(vial_detections, start=1):

                vial_bbox = vial_detection["bbox"]

                # Find metadata for this specific vial
                matches = metadata[
                    (metadata["image"] == image_name) &
                    (metadata["vial_ind"] == vial_index)
                ]

                if matches.empty:
                    print("\n⚠️ NO METADATA MATCH FOUND")
                    print("Image:", image_name)
                    print("Vial index:", vial_index)
                    print("Available metadata for this image:")
                    print(metadata[metadata["image"] == image_name])

                    continue

                metadata_row = matches.iloc[0]

                # Create Vial with formulation metadata
                vial = Vial(
                    vial_ind=vial_index,

                    sample_id=metadata_row["sample_id"],
                    surfactant=metadata_row["surfactant"],
                    surfactant_conc=metadata_row["surfactant_conc"],
                    oil=metadata_row["oil"],
                    oil_conc=metadata_row["oil_conc"],
                    pH=metadata_row["pH"],
                    time_s=metadata_row["time_s"],

                    vial_conf=vial_detection["confidence"],
                    vial_bbox_x1y1_x2y2=vial_bbox
                )

                # Look at every phase detection
                for phase_detection in phase_detections:

                    phase_bbox = phase_detection["bbox"]

                    # Find centre of phase bounding box
                    phase_center = box_center(phase_bbox)

                    # Check whether phase is inside vial
                    if point_inside_box(phase_center, vial_bbox):

                        phase = Phase(
                            phase_ind=phase_detection["class_ind"],
                            phase_name=phase_detection["class_name"],
                            phase_conf=phase_detection["confidence"],
                            phase_bbox_x1y1_x2y2=phase_bbox
                        )

                        vial.phases.append(phase)

                # Add vial to ImageResult
                image_result.vials.append(vial)

        # Print results

        print("\nIMAGE RESULT")
        print("=" * 50)

        print("Filename:", image_result.filename)
        print("Number of vials:", len(image_result.vials))

        for vial in image_result.vials:

            print("\nVial", vial.vial_ind)
            print(f"Confidence: {vial.vial_conf * 100:.2f}%")
            print("Bounding box:", vial.vial_bbox_x1y1_x2y2)

            print("Phases:")

            for phase in vial.phases:

                print(
                    f"  {phase.phase_name}: "
                    f"{phase.phase_conf * 100:.2f}%"
                )
    
        # Add this image to the overall results
        all_image_results.append(image_result)
    
    # Save results
    save_json(
        all_image_results,
        "LF_RS_pH5_detection4.json"
    )

    save_csv(
        all_image_results,
        "LF_RS_pH5_detection4.csv"
    )

if __name__ == "__main__":
    main()