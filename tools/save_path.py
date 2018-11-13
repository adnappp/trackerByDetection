import pandas as pd
import numpy as np




def save_excel(results,log_file,CameraId):
    log = pd.read_csv(log_file)
    time = log['Time']
    time.append(str(i / 25))
    for d in results:
        d = d.astype(np.int32)
        cowid = d[4]
        if str(cowid)
            col = log[str(cowid)]
        data = str(d[0] + d[2] / 2) + ',' + str(d[1] + d[3] / 2) + ',' + str(CameraId)
        col = col.append(pd.Series(data))
    log.to_csv(log_file)