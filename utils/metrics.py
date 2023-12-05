#用于评估图像和文本之间的匹配性能
import numpy as np

def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #scores_i2t: 一个数组，其中包含图像到文本匹配得分。
    #scores_t2i: 一个数组，其中包含文本到图像匹配得分。
    #txt2img: 一个数组或列表，表示正确的文本到图像的匹配索引。
    #img2txt: 一个数组或列表，表示正确的图像到文本的匹配索引。
    # Images->Text 从图像到文本的匹配

    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        #对当前的 score 数组进行降序排序，并获取排序后的索引。
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result