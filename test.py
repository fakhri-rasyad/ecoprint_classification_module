from roboflow import Roboflow
rf = Roboflow(api_key="h2WcgukjZNOxkMDxxwth")
project = rf.workspace("p-dtulo").project("fabric_datasett-kkvd4")
version = project.version(4)
dataset = version.download("yolov5")