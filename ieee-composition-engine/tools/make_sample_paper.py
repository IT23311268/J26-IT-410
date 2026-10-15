"""
make_sample_paper.py - builds a deliberately messy test paper.

You need something to feed the engine before you have a real paper handy, and
you need it to be messy in the exact ways the engine is supposed to fix:

  * headings named by the author's whim ("Background", "Our Approach",
    "Final Remarks") instead of IEEE's names
  * inconsistent heading styles - some numbered, some not
  * references in APA style instead of IEEE's [1] style
  * in-text citations as (Author, year) instead of [1]
  * a running header and page numbers that must be thrown away
  * single column, which IEEE is not

Run it once:
    python tools/make_sample_paper.py

It writes data/input/sample_paper.pdf.
"""

from __future__ import annotations

import os

import pymupdf

PAGE_WIDTH, PAGE_HEIGHT = 595, 842      # A4 in points
MARGIN = 65
BODY_SIZE = 10.5

TIMES = "tiro"          # Times-Roman
TIMES_BOLD = "tibo"     # Times-Bold
TIMES_ITALIC = "tiit"   # Times-Italic

ALIGN_LEFT, ALIGN_CENTER, ALIGN_JUSTIFY = 0, 1, 3

# (text, font_size, font_name, alignment, space_after_in_points)
CONTENT = [
    ("Automated Detection of Pulmonary Nodules in Chest CT Images Using "
     "Convolutional Neural Networks", 16, TIMES_BOLD, ALIGN_CENTER, 14),

    ("K. M. Perera, S. Fernando and A. Jayasuriya", 11, TIMES, ALIGN_CENTER, 4),
    ("Department of Computer Science, University of Colombo, Sri Lanka",
     9.5, TIMES_ITALIC, ALIGN_CENTER, 18),

    ("ABSTRACT", 11, TIMES_BOLD, ALIGN_LEFT, 6),
    ("Lung cancer remains one of the leading causes of cancer related death "
     "worldwide, and early detection of pulmonary nodules substantially "
     "improves patient survival. Manual inspection of computed tomography "
     "volumes is slow and suffers from inter observer variability. In this "
     "work we present a convolutional neural network that detects pulmonary "
     "nodules in chest CT slices. The model was trained on 888 annotated "
     "scans and reached a sensitivity of 94.2 percent at 2.1 false positives "
     "per scan, outperforming the three baseline methods we compared against.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 10),

    ("Keywords: lung cancer, computed tomography, deep learning, nodule "
     "detection, medical imaging", BODY_SIZE, TIMES_ITALIC, ALIGN_LEFT, 16),

    ("1. Background", 12, TIMES_BOLD, ALIGN_LEFT, 6),
    ("Lung cancer accounted for approximately 1.8 million deaths in 2020 "
     "(Sung et al., 2021). Screening programmes based on low dose computed "
     "tomography have been shown to reduce mortality by up to 20 percent, but "
     "they generate an enormous volume of images that radiologists must read. "
     "A single chest CT study can contain more than 300 slices, and a nodule "
     "may occupy only a handful of pixels in one of them.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 8),
    ("Computer aided detection systems were introduced to reduce this burden. "
     "Early systems relied on hand engineered features such as shape "
     "compactness and intensity thresholds, which generalised poorly across "
     "scanners (Firmino et al., 2016). The arrival of deep learning changed "
     "this picture, since features are learned directly from data.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 14),

    ("2. Previous Studies", 12, TIMES_BOLD, ALIGN_LEFT, 6),
    ("Setio et al. (2016) proposed a multi view convolutional network that "
     "aggregates predictions from nine planes through a fixed fusion rule, "
     "reporting a sensitivity of 90.1 percent. Dou et al. (2017) extended "
     "this idea with three dimensional convolutions and an online hard "
     "negative mining strategy. More recently, attention based architectures "
     "have been applied to the same task with mixed results, and several "
     "authors note that reported figures are difficult to compare because "
     "evaluation protocols differ between studies.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 14),

    ("3. Our Approach", 12, TIMES_BOLD, ALIGN_LEFT, 6),

    ("3.1 Dataset", 11, TIMES_BOLD, ALIGN_LEFT, 5),
    ("We used the LUNA16 collection, which contains 888 thoracic CT scans "
     "with nodule annotations agreed by at least three of four radiologists. "
     "Scans were resampled to an isotropic spacing of 1 mm and intensities "
     "were clipped to the range -1000 to 400 Hounsfield units before being "
     "rescaled to the interval zero to one.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 8),

    ("3.2 Network Architecture", 11, TIMES_BOLD, ALIGN_LEFT, 5),
    ("The detector follows an encoder decoder design with five downsampling "
     "stages. Each stage applies two three dimensional convolutions with "
     "batch normalisation and a rectified linear activation. The loss "
     "function combines a Dice term with a focal term in order to cope with "
     "the severe class imbalance between nodule and background voxels:",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 8),
    ("L = alpha * L_dice + (1 - alpha) * L_focal          (1)",
     BODY_SIZE, TIMES_ITALIC, ALIGN_CENTER, 8),
    ("where alpha was fixed at 0.7 after a small grid search on the "
     "validation fold.", BODY_SIZE, TIMES, ALIGN_JUSTIFY, 8),
    ("Figure 1. Overall architecture of the proposed detection network.",
     9.5, TIMES, ALIGN_CENTER, 14),

    ("4. Experiments and Findings", 12, TIMES_BOLD, ALIGN_LEFT, 6),
    ("Training used the Adam optimiser with an initial learning rate of "
     "0.001, halved whenever validation loss failed to improve for five "
     "epochs. We report sensitivity at a fixed operating point of two false "
     "positives per scan, following the convention used by the LUNA16 "
     "challenge. The proposed model reached 94.2 percent sensitivity, "
     "compared with 90.1 percent for the multi view baseline and 92.0 "
     "percent for the three dimensional baseline.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 8),
    ("Table 1: Comparison of detection performance on the test fold.",
     9.5, TIMES, ALIGN_CENTER, 8),
    ("An ablation study confirmed that the focal term contributed roughly "
     "1.8 percentage points of the improvement, while isotropic resampling "
     "contributed a further 0.9 points.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 14),

    ("5. Final Remarks", 12, TIMES_BOLD, ALIGN_LEFT, 6),
    ("We presented a three dimensional convolutional detector for pulmonary "
     "nodules that improves sensitivity over established baselines while "
     "keeping the false positive rate low. The main limitation is that "
     "evaluation was confined to a single public dataset acquired in a narrow "
     "range of scanners. Future work will examine cross institution "
     "generalisation and the calibration of predicted confidence scores.",
     BODY_SIZE, TIMES, ALIGN_JUSTIFY, 14),

    ("References", 12, TIMES_BOLD, ALIGN_LEFT, 6),
    ("Dou, Q., Chen, H., Jin, Y., Lin, H., Qin, J., & Heng, P. A. (2017). "
     "Automated pulmonary nodule detection via 3D ConvNets with online sample "
     "filtering. Medical Image Computing and Computer Assisted Intervention, "
     "559-567.", 9.5, TIMES, ALIGN_LEFT, 5),
    ("Firmino, M., Angelo, G., Morais, H., Dantas, M. R., & Valentim, R. "
     "(2016). Computer aided detection and diagnosis of lung cancer: a review. "
     "BioMedical Engineering OnLine, 15(1), 2.", 9.5, TIMES, ALIGN_LEFT, 5),
    ("Setio, A. A. A., Ciompi, F., Litjens, G., Gerke, P., Jacobs, C., & van "
     "Riel, S. J. (2016). Pulmonary nodule detection in CT images using "
     "multi view convolutional networks. IEEE Transactions on Medical "
     "Imaging, 35(5), 1160-1169.", 9.5, TIMES, ALIGN_LEFT, 5),
    ("Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., "
     "& Jemal, A. (2021). Global cancer statistics 2020. CA: A Cancer Journal "
     "for Clinicians, 71(3), 209-249.", 9.5, TIMES, ALIGN_LEFT, 5),
]

RUNNING_HEADER = "Preprint - not peer reviewed"


def new_page(document: pymupdf.Document) -> tuple:
    """Add a page, draw the running header, and return (page, start_y)."""
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text(
        (MARGIN, 40), RUNNING_HEADER, fontsize=8, fontname=TIMES_ITALIC,
    )
    return page, 58


def write_block(page, cursor_y: float, text: str, size: float,
                font: str, align: int) -> float:
    """Draw one block of text and return how tall it turned out to be."""
    box = pymupdf.Rect(MARGIN, cursor_y, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 55)
    leftover = page.insert_textbox(
        box, text, fontsize=size, fontname=font, align=align, lineheight=1.35,
    )
    if leftover < 0:
        return -1.0          # did not fit on this page
    return box.height - leftover


def build(output_path: str) -> None:
    document = pymupdf.open()
    page, cursor_y = new_page(document)
    page_number = 1

    for text, size, font, align, space_after in CONTENT:
        used = write_block(page, cursor_y, text, size, font, align)

        if used < 0:                       # ran off the bottom - start a page
            page.insert_text(
                (PAGE_WIDTH / 2 - 6, PAGE_HEIGHT - 40), str(page_number),
                fontsize=9, fontname=TIMES,
            )
            page, cursor_y = new_page(document)
            page_number += 1
            used = write_block(page, cursor_y, text, size, font, align)

        cursor_y += used + space_after

    page.insert_text(
        (PAGE_WIDTH / 2 - 6, PAGE_HEIGHT - 40), str(page_number),
        fontsize=9, fontname=TIMES,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    document.save(output_path)
    document.close()
    print(f"Sample paper written to: {output_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(here, "data", "input", "sample_paper.pdf"))
