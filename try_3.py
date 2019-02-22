import matplotlib.pyplot as plt
import wfdb
from wfdb import processing


sig, fields=wfdb.rdsamp('100', channels=[0], sampto=15000, pb_dir='mitdb/')
ann_ref=wfdb.rdann('100', 'atr', sampto=15000, pb_dir='mitdb/')

#使用XQRS算法

xqrs=processing.XQRS(sig=sig[:,0], fs=fields['fs'])
xqrs.detect()

#这里还可以直接使用xqrs_detection
#qrs_inds=processing.xqrs_detect(sig=sig[:,0], fs=fields['fs'])

#下面进行算法的结果和注释中的结果相对比
#注意：在100.atr中的qrs注释第一个数是18，而18这个位置上并没有峰值，真正的第一个峰值是在第二个数77开始的所以是[1:]

comparitor=processing.compare_annotations(ref_sample=ann_ref.sample[1:],
                                          test_sample=xqrs.qrs_inds,
                                          window_width=int(0.1*fields['fs']),
                                          signal=sig[:,0])

#输出结果
comparitor.print_summary()
fig=comparitor.plot(title='XQRS detected QRS vs reference annotations',return_fig=True)
# display(fig[0])
plt.show(fig[0])#这一步必须加，不然图片会一闪而逝
