from os.path import isfile, join, splitext, basename
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

class GFG:
    def __init__(self,graph):
        self.graph = graph 
        self.gt = len(graph)
        self.predict = len(graph[0])
  
    def bpm(self, u, matchR, seen):
        for v in range(self.predict):
            if self.graph[u][v] and seen[v] == False:
                seen[v] = True 
                if matchR[v] == -1 or self.bpm(matchR[v], 
                                               matchR, seen):
                    matchR[v] = u
                    return True
        return False
  
    def maxBPM(self):
        matchR = [-1] * self.predict
          
        result = 0 
        for i in range(self.gt):
            seen = [False] * self.predict
              
            if self.bpm(i, matchR, seen):
                result += 1
        return result, matchR

def cal_F1(groudTruth_csvFilePath, predict_csvFilePath):
    # print(groudTruth_csvFilePath)
    checkDistance = 20 ** 2
    
    gt = pd.read_csv(groudTruth_csvFilePath, header=None).to_numpy()
    predict = pd.read_csv(predict_csvFilePath, header=None).to_numpy()
    positiveCount = len(gt)
    predictCount = len(predict)
    

    graph = np.sum((gt[:,np.newaxis,:] - predict[np.newaxis,:,:]) ** 2, axis=-1) <= checkDistance
    g = GFG(graph)
    matchCount, matchR = g.maxBPM()

    P = matchCount / predictCount
    R = matchCount / positiveCount
    F = 2 * P * R / (P + R)
    return P, R, F

def main(opt):
    n_gts = len(list(opt.gts_d.glob('*.csv')))
    data = []
    with tqdm(opt.gts_d.glob('*.csv'), total=n_gts) as itemlist:
        for i, gt_p in enumerate(itemlist):
            predict_p = opt.predicts_d / gt_p.name
            p, r, f = cal_F1(str(gt_p), str(predict_p))
            data.append((f, p, r, gt_p.name))
    
    df = pd.DataFrame(data, columns=('F1', 'Precision', 'Recall', 'path'))
    # df = df.astype({
    #     'F1': np.float64,
    #     'Precision': np.float64,
    #     'Recall': np.float64,
    #     'path': str,
    # })
    df = df.sort_values(by=['F1', 'Precision', 'Recall'])
    print(df)
    f, p, r = np.average(df.iloc[:, 0:3], axis=0)
    print("F1:", f)
    print("Precision:", p)
    print("Recall:", r)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gts_d', type=str, help='Directory of ground truth csv files')
    parser.add_argument('predicts_d', type=str, help='Directory of predicted csv files')
    opt = parser.parse_args()
    opt.gts_d = Path(opt.gts_d)
    opt.predicts_d = Path(opt.predicts_d)
    main(opt)