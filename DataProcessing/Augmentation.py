import ffmpy
import os


def change_tempo(currPath, newPath, tempo):
    ff = ffmpy.FFmpeg(
        inputs={currPath: None},
        outputs={newPath: ["-filter:a", "atempo=" + str(tempo)]},
    )
    ff.run()
    return


def aug_tempo(pathsToFolders, initialNumber, tempo):
    for path in pathsToFolders:
        files = sorted(os.listdir(path))[:100]
        for i, file in enumerate(files):
            fileSplit = file.split(".")
            currPath = path + "/" + file
            newPath = (
                path + "/" + fileSplit[0] + str(i + initialNumber).zfill(5) + ".wav"
            )
            change_tempo(currPath, newPath, tempo)
    return


if __name__ == "__main__":
    keyPath = "./Data/genres/"
    appends = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    pathsToFolders = [keyPath + e for e in appends]
    aug_tempo(pathsToFolders, initialNumber=100, tempo=1.1)
    aug_tempo(pathsToFolders, initialNumber=200, tempo=0.9)
