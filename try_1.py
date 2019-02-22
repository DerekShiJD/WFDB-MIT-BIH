import wfdb
import matplotlib.pyplot as plt

'''Part1 read signal and plot signal'''

signal, fields = wfdb.rdsamp('100',channels=[0, 1], sampfrom=0, sampto=1500, pb_dir='mitdb/')

print('signal:',signal)
print('fields:',fields)
plt.plot(signal)
plt.ylabel(fields['units'][0])
plt.legend([fields['sig_name'][0],fields['sig_name'][1]])
plt.show()


'''Part2 annotation and record'''

record = wfdb.rdrecord('data/mitdb/100', sampto=3600)
annotation = wfdb.rdann('data/mitdb/100', 'atr', sampto=3600)
wfdb.plot_wfdb(record=record, annotation=annotation,
               title='Record 100 from MIT-BIH Arrhythmia Database', time_units='seconds')



print('annotation:', annotation.__dict__)
print('anno positions:', annotation.sample)
print('anno labels:', annotation.symbol)
print('record:', record.__dict__)
print('signal:', record.p_signal)

wfdb.show_ann_labels()