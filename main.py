import cv2
from detector import Detector
from enum import Enum


class Color(Enum):  # BGR
    GREEN = (0, 255, 0)
    LIGHT_GREEN = (30, 255, 30)
    LIME = (70, 200, 70)
    RED = (30, 30, 255)
    BLACK = (0, 0, 0)
    GREY = (200, 200, 200)
    WHITE = (255, 255, 255)
    BLUE = (250, 140, 40)
    GOLD = (20, 220, 240)
    MAGENTA = (255, 50, 255)
    PURPLE = (128, 50, 128)


def main(source, nth_frame=3):
    try:
        detector = Detector()
        input_video = cv2.VideoCapture(source)
        current_frame = 0
        while True:
            ret, frame = input_video.read()
            if not ret:
                input_video = cv2.VideoCapture(source)
                current_frame = 0
                continue
            if current_frame % nth_frame == 0:
                results = detector.find_all(frame)
                for result in results:
                    if result["label"] == 0:
                        color = Color.PURPLE.value
                        label = "Bike"
                    elif result["label"] == 1:
                        color = Color.RED.value
                        label = "Person"
                    else:
                        color = Color.GREEN.value
                        label = "Vehicle"
                    cv2.rectangle(frame, (result["xmin"], result["ymin"]), (result["xmax"], result["ymax"]),
                                  color, thickness=1)

                    cv2.putText(frame, "%s: %.2f" % (label, round(result["confidence"], 2)), (result["xmin"], result["ymax"] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            current_frame += 1

    except Exception as e:
        print(e)


rtsp = 'rtsp://iStream10-echd.mos.ru:9554/GFLSYPCN752UYKZXWDEKNI53NMNIY3ZY333XX3E5E3HN7MVHBAUJK2UMVVY634UOZQEMUNK4KJDJNBVYIXAYCNACOI6BCO37IQF3OP7MTSGN6ZU2ETJ2GPY3HW7B2MMQCZHNJCFMHRV72ZOZZVSFJPUVSP7QPGVDO3TSWQBF7ZWP4H5FYWU7DEYPIW6KXBTJ2UDZXVYNUVS5L7RRQ65TK7UKES2TB25XZCLDDQ4ZUQMMMFX3S4FWP7LY6U6T3RRRFY4YDHUGON6YJDFHI2TH62XOCK54PW6VJPLT74EFF457NJNETCWEZZJ5QX5E3IEDG7NYOMRFFHSP6GE7NEUF2XBQZOHMC5UL2C3XA4QSJWTUF6ADQROMYC3IWDGMKU4R2PDAGCWQNIUXPDBZEYCFGAQFPKAYJRBR5UVKYFKIAZVZS4M7SD3EL6JN2QORW3EF4UWS6GFPSPQYGTFEYV2RCOUX3YNHPDY4U5AT62A/a43956f9d802ab6dbc5056959d14b0ad-public'
rtsp2 = 'rtsp://iStream9-echd.mos.ru:8554/L23MFY36JEJ5GN2M365HXK7GH5UXSVVDUJQP3BO4DCKVMM2N3R2DU6PSLPGG6CDZYMHX7S3CPWQTA6EOFDM55JJYBPA3J6CPWRIEI3ZECZYEFE27TFL6OJ4SFG24BXRQVSDP6TNRXARZUB6MFTMIYOUDVRHD4OMTK6Y2ZLQYU2LPUWLFWLC2EHQQZ2NDISVYBI4ZKM6P5GJOK/d2a09f4e161a590f158b456744bb194f-public'
main(rtsp2)
