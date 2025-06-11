import numpy as np
import matplotlib.pyplot as plt
import h5py
from airPLS.airPLS import airPLS
from sklearn.linear_model import LinearRegression
import pickle 

class PhotometryData: 
    def __init__(self, f):
        self.filename = f
        self.data = h5py.File(f) 
        self.CAM1 = self.data['DataAcquisition']['BFPD']['ROIs']['Series0001']['CAM1_EXC1']
        self.CAM2 = self.data['DataAcquisition']['BFPD']['ROIs']['Series0001']['CAM1_EXC2']
        self.data1 = self.unroll_CAM_data(self.CAM1)
        self.time1 = self.unroll_CAM_time(self.CAM1)
        self.data2 = self.unroll_CAM_data(self.CAM2)
        self.time2 = self.unroll_CAM_time(self.CAM2)
        
    def unroll_CAM_data(self, cam): 
        keys = cam.keys()
        ROIs = [key for key in keys if 'ROI' in key]
        data = np.zeros((len(ROIs), len(cam['Time']))) 
        for i, roi in enumerate(ROIs):
            data[i, :] = np.array(cam[roi])
        return data

    def unroll_CAM_time(self, cam):
        return np.array(cam['Time'])

    def truncate_data(self): 
        stop = min([self.data1.shape[1], self.data2.shape[1]]) 
        self.data1 = self.data1[:, :stop]
        self.data2 = self.data2[:, :stop] 
        self.time1 = self.time1[:stop]
        self.time2 = self.time2[:stop]
        
    def add_data(self, time1, data1, time2, data2, fig, ax1, ax2, stop=None): 
        if not stop: 
            stop = max([len(self.time1), len(self.time2)])
        ax1.plot(time1, data1, alpha=0.7, label='cam1') 
        ax1.plot(time2, data2, alpha=0.7, label='cam2')
        ax1.set_title('full time course')
        ax2.plot(time1[:stop], data1[:stop], alpha=0.7) 
        ax2.plot(time2[:stop], data2[:stop], alpha=0.7)
        ax2.set_title('short time course') 
    
    def plot_raw_data(self, roi=0, stop=600): 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        self.add_data(self.time1, self.data1[roi, :], self.time2, self.data2[roi, :], fig, ax1, ax2, stop) 
        ax1.legend()
        fig.suptitle(f'ROI {roi}')
        fig.savefig(f'Plots/dopamine_raw_data_roi_{roi}.png') 
        plt.close(fig)
    
    def plot_airPLS_corrected_data(self, roi=0, stop=600): 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        self.add_data(self.time1, self.data1_airPLS[roi, :], self.time2, self.data2_airPLS[roi, :], fig, ax1, ax2, stop) 
        ax1.legend()
        fig.suptitle(f'ROI {roi} (baseline corrected)')
        fig.savefig(f'Plots/dopamine_baseline_corrected_roi_{roi}.png') 
        plt.close(fig)
    
    def plot_zscored_data(self, roi=0, stop=600):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        self.add_data(self.time1, self.zscore1[roi, :], self.time2, self.zscore2[roi, :], fig, ax1, ax2, stop) 
        ax1.legend()
        fig.suptitle(f'ROI {roi} (z-scored)')
        fig.savefig(f'Plots/dopamine_z_scored_roi_{roi}.png') 
        plt.close(fig)
    
    def plot_regressed_data(self, roi=0, stop=600):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.plot(self.time1, self.regressed1[roi, :], label='regressed signal') 
        ax1.set_title('full time course') 
        ax2.plot(self.time1[:stop], self.regressed1[roi, :stop])
        ax2.set_title('short time course')
        ax1.legend()
        fig.suptitle(f'ROI {roi} (z-scored)')
        fig.savefig(f'Plots/dopamine_regressed_roi_{roi}.png') 
        plt.close(fig)
    
    def plot_mean_smoothed(self, i, roi=0, window=5, stop=100): 
        self.smoothed_time = self.smoothed_time[:len(self.mean_smoothed[1])]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.plot(self.smoothed_time, self.mean_smoothed[roi, :], label=f'mean smoothed (window={window}') 
        ax1.set_title('full time course') 
        ax2.plot(self.smoothed_time[:stop], self.mean_smoothed[roi, :stop])
        ax2.set_title('short time course')
        ax1.legend()
        fig.suptitle(f'ROI {roi} (mean smoothed)')
        fig.savefig(f'Plots/dopamine_mean_smoothed_roi_{roi}_{i}.png') 
        plt.close(fig)

    def airPLS_correct_data(self, data, lambda_, porder, itermax): 
        corrected_data = np.zeros(data.shape)
        for roi in range(len(data)): 
            background = airPLS(data[roi], lambda_, porder, itermax)
            corrected_data[roi] = data[roi] - background
        return corrected_data
    
    def correct_using_airPLS(self, lambda_=100, porder=1, itermax=15): 
        self.data1_airPLS = self.airPLS_correct_data(self.data1, lambda_, porder, itermax)
        self.data2_airPLS = self.airPLS_correct_data(self.data2, lambda_, porder, itermax)
        
    def z_score_data(self, data, session_medians=None, session_std_devs=None):
        '''session medians and std_devs should be a 1d array of length == number of rois, one median per roi''' 
        ''' if no session medians are provided, we use the trial median--this is just a placeholder for now''' 
        if not session_medians:
            session_medians = np.median(data, axis=1) 
            session_std_devs = np.std(data, axis=1) 
        return ((data.T - session_medians) / session_std_devs).T
    
    def z_score(self, session_medians1=None, session_std_devs1=None, session_medians2=None, session_std_devs2=None): 
        self.zscore1 = self.z_score_data(self.data1_airPLS, session_medians1, session_std_devs1) 
        self.zscore2 = self.z_score_data(self.data2_airPLS, session_medians2, session_std_devs2)
        
    def regress_ref_onto_signal(self): 
        num_rois = len(self.zscore1)  # Number of ROIs
        min_length = min([len(self.zscore1[roi]) for roi in range(num_rois)] + 
                        [len(self.zscore2[roi]) for roi in range(num_rois)])
        self.regressed1 = np.zeros((num_rois, min_length))
        for roi in range(num_rois):
            reg = LinearRegression(positive=True)
            x_data = self.zscore1[roi][:min_length]
            y_data = self.zscore2[roi][:min_length]
            fitted = reg.fit(x_data.reshape(-1, 1), y_data).predict(x_data.reshape(-1, 1))
            self.regressed1[roi] = x_data - fitted
            
    def downsample_and_smooth(self, window=5, padding=2, sampling_rate=None):
        # match lengths of time1 and regressed1
        min_length = min(len(self.time1), self.regressed1.shape[1])
        self.time1 = self.time1[:min_length]
        self.regressed1 = self.regressed1[:min_length]

        # todo incorporate downsampling, not sure how you do that! 
        self.mean_smoothed = np.zeros((self.regressed1.shape[0], self.regressed1.shape[1] - 2 * padding))
        for roi in range(len(self.regressed1)): 
            self.mean_smoothed[roi] = np.convolve(self.regressed1[roi], np.ones(window)/window, mode='valid')
        self.smoothed_time = self.time1[padding:-padding]

        
    def run_full_processing(self): 
        pass
    
    def plot_full_processing(self): 
        pass

    def store_preprocessed_data(self):
        self.processed_data = {
            "filename": self.filename,
            "time1": self.time1,
            "time2": self.time2,
            "data1_airPLS": self.data1_airPLS,
            "data2_airPLS": self.data2_airPLS,
            "zscore1": self.zscore1,
            "zscore2": self.zscore2,
            "regressed1": self.regressed1,
            "mean_smoothed": self.mean_smoothed,
            "smoothed_time": self.smoothed_time,
        }

def main():
    list = ['Data/DAT_6_0000.doric', 'Data/DAT_C1_0008.doric', 'Data/DAT_C2_0009.doric', 'Data/DAT_C4_0010.doric', 'Data/DAT_C5_0011.doric']

    processed_data_dict = {}

    for i, file in enumerate(list):
        pt = PhotometryData(file)
        
        # preprocess
        pt.correct_using_airPLS()
        pt.z_score()
        pt.regress_ref_onto_signal()
        pt.downsample_and_smooth()

        # plot
        for roi in range(19): # 19 ROIs in each
            """pt.plot_raw_data(roi=roi)
            pt.plot_airPLS_corrected_data(roi=roi)
            pt.plot_zscored_data(roi=roi)
            pt.plot_regressed_data(roi=roi)"""
            pt.plot_mean_smoothed(i=i, roi=roi)
        
        # store
        pt.store_preprocessed_data()
        processed_data_dict[file] = pt.processed_data

    with open("Data/photometry_processed_all.pkl","wb") as f:
        pickle.dump(processed_data_dict,f)

if __name__ == "__main__":
    main()

