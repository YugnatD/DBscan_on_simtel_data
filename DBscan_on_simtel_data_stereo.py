#!/home/burmist/miniconda/envs/eventio/bin/python
# -*- coding: utf-8 -*-

#hostname dpnc02
#/home/burmist/miniconda/envs/eventio/bin/python
#hostname daint
#/users/lburmist/miniconda/envs/pyeventio/bin/python

from eventio import SimTelFile
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#
#
_n_max_noise_events=1000
_npe_noise=20
_n_of_time_sample=75
_time_of_one_sample_s=(_n_of_time_sample*1000/1024.0*1.0e-9)
_event_modulo=1
#
_time_norm=0.05
_DBSCAN_digitalsum_threshold = 2150
_DBSCAN_eps = 0.2
_DBSCAN_min_samples = 15
#
#

def print_setup():
    print("_n_max_noise_events          = ",_n_max_noise_events)
    print("_npe_noise                   = ",_npe_noise)
    print("_time_of_one_sample_s        = ",_time_of_one_sample_s)
    print("_event_modulo                = ",_event_modulo)
    print("_time_norm                   = ",_time_norm)
    print("_DBSCAN_digitalsum_threshold = ",_DBSCAN_digitalsum_threshold)
    print("_DBSCAN_eps                  = ", _DBSCAN_eps)
    print("_DBSCAN_min_samples          = ",_DBSCAN_min_samples)
    
def extend_pixel_mapping( pixel_mapping, channel_list, number_of_wf_time_samples):
    pixel_mapping_extended=pixel_mapping[channel_list[:,0]].copy()
    pixel_mapping_extended=pixel_mapping_extended[:,:-1]
    pixel_mapping_extended=np.expand_dims(pixel_mapping_extended,axis=2)
    pixel_mapping_extended=np.swapaxes(pixel_mapping_extended, 1, 2)
    pixel_mapping_extended=np.concatenate(([pixel_mapping_extended for i in np.arange(0,number_of_wf_time_samples)]), axis=1)
    pixt=np.array([i for i in np.arange(0,number_of_wf_time_samples)]).reshape(1,number_of_wf_time_samples)
    pixt=np.concatenate(([pixt for i in np.arange(0,channel_list.shape[0])]), axis=0)
    pixt=np.expand_dims(pixt,axis=2)
    pixel_mapping_extended=np.concatenate((pixel_mapping_extended,pixt), axis=2)
    #print(pixel_mapping_extended)
    #print(pixel_mapping_extended.shape)
    return pixel_mapping_extended

def get_DBSCAN_clusters( digitalsum, pixel_mapping, pixel_mapping_extended, channel_list, time_norm, digitalsum_threshold, DBSCAN_eps, DBSCAN_min_samples):
    #extend_pixel_mapping( pixel_mapping=pixel_mapping, channel_list=channel_list, number_of_wf_time_samples=digitalsum.shape[1])
    #print(digitalsum.shape)
    #print(pixel_mapping.shape)
    #print(print(pixel_mapping_extended.shape)
    #
    clusters_info=def_clusters_info()
    X=pixel_mapping_extended[digitalsum>digitalsum_threshold] 
    X=X*[[1,1,time_norm]]
    dbscan = DBSCAN( eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
    clusters = dbscan.fit_predict(X)
    pointID = np.unique(clusters)
    print(pointID)
    
    #time_norm_arr=np.array([[1,1,time_norm]])
    #time_norm_arr=np.concatenate(([time_norm_arr for i in np.arange(0,points.shape[0])]), axis=0)
    #print(time_norm_arr.shape)
    #print(points.shape)
    #print(time_norm_arr)
    #print(points)
    #print()
    #if (pixel_mapping_extended.shape[0]>0):

    #print(pixel_mapping_extended[:,:,2]*)
    #print(channel_list[:,0])
    #pixel_mapping_filtered=pixel_mapping[channel_list[:,0]].copy()
    #print(pixel_mapping_filtered.shape)
    #
    #pix_x=pixel_mapping_extended[:,0].reshape(pixel_mapping_extended.shape[0],1)
    #pix_y=pixel_mapping_extended[:,1].reshape(pixel_mapping_extended.shape[0],1)
    #pix_x=np.concatenate(([pix_x for i in np.arange(0,digitalsum.shape[1])]), axis=1)
    #pix_y=np.concatenate(([pix_y for i in np.arange(0,digitalsum.shape[1])]), axis=1)
    #
    #pix_t=np.array([i for i in np.arange(0,digitalsum.shape[1])]).reshape(1,digitalsum.shape[1])
    #pix_t=pix_t*time_norm
    #pix_t=np.concatenate(([pix_t for i in np.arange(0,digitalsum.shape[0])]), axis=0)
    ##
    #pix_x=pix_x[digitalsum>digitalsum_threshold]
    #pix_y=pix_y[digitalsum>digitalsum_threshold]
    #pix_t=pix_t[digitalsum>digitalsum_threshold]  
    ##
    #pix_x=np.expand_dims(pix_x,axis=1)
    #pix_y=np.expand_dims(pix_y,axis=1)
    #pix_t=np.expand_dims(pix_t,axis=1)
    #
    #print(pix_x.shape)
    #print(pix_y.shape)
    #print(pix_t.shape)
    #
    #X=np.concatenate((pix_x,pix_y,pix_t), axis=1)
    #dbscan = DBSCAN( eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
    #clusters = dbscan.fit_predict(X)
    #pointID = np.unique(clusters)
    ##
    #
    #clusters_info['n_digitalsum_points'] = len(pix_x)
    #
    #if len(pointID) > 1 :
    #    pointID = pointID[pointID>-1]
        #print(pointID)
    #    clustersID = np.argmax([len(clusters[clusters==clID]) for clID in pointID])            
        #
    #    clusters_info['n_clusters'] = len(pointID)
    #    clusters_info['n_points'] = len(clusters[clusters==clustersID])
    #    clusters_info['x_mean'] = np.mean(pix_x[clusters==clustersID])
    #    clusters_info['y_mean'] = np.mean(pix_y[clusters==clustersID])
    #    clusters_info['t_mean'] = np.mean(pix_t[clusters==clustersID])
        #
    #    clusters_info['channelID'] = get_channelID_from_x_y( pixel_mapping_extended=pixel_mapping_extended, x_val=clusters_info['x_mean'], y_val=clusters_info['y_mean'])
    #    clusters_info['timeID'] = get_timeID( number_of_time_points=digitalsum.shape[1], time_norm=time_norm, t_val=clusters_info['t_mean'])
        #
        #clustern = np.max([ for clID in np.unique(clusters)[1:]])
        #clustern = 0
        #print(len(cl_ID))
        #print(clustern)
    #
    return clusters_info
    
def digital_w_sum( wf, digi_sum_channel_list, channels_blacklist):
    #
    digital_sum_result=None
    for i in np.arange(0,len(digi_sum_channel_list)):
        not_contains_any=np.sum(np.isin(digi_sum_channel_list[i],channels_blacklist))
        if (not_contains_any == 0):
            if digital_sum_result is None:
                digital_sum_result=np.sum(wf[digi_sum_channel_list[i]],axis=0)
                digital_sum_result=np.reshape(digital_sum_result,(1,len(digital_sum_result)))
            else:
                digital_sum_result_tmp=np.sum(wf[digi_sum_channel_list[i]],axis=0)
                digital_sum_result_tmp=np.reshape(digital_sum_result_tmp,(1,len(digital_sum_result_tmp)))
                digital_sum_result=np.concatenate((digital_sum_result,digital_sum_result_tmp))
                #
    return digital_sum_result
                
def digital_sum( wf, digi_sum_channel_list):
    digital_sum_result = np.array([np.sum(wf[digi_sum_channel_list[i]],axis=0) for i in np.arange(0,len(digi_sum_channel_list))])
    return digital_sum_result

def def_clusters_info( n_digitalsum_points=0, n_clusters=0, n_points=0,
                       x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                       channelID=-999, timeID=-999):
    #
    clusters_info={'n_digitalsum_points':n_digitalsum_points,
                   'n_clusters':n_clusters,
                   'n_points':n_points,
                   'x_mean':x_mean,
                   'y_mean':y_mean,
                   't_mean':t_mean,
                   'channelID':channelID,
                   'timeID':timeID}
    #
    return clusters_info

def def_L1_trigger_info( max_digi_sum=0,
                         x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                         channelID=-999, timeID=-999):
    L1_trigger_info={'max_digi_sum':max_digi_sum,
                     'x_mean':x_mean,
                     'y_mean':y_mean,
                     't_mean':t_mean,
                     'channelID':channelID,
                     'timeID':timeID}
    #
    return L1_trigger_info

def get_L1_trigger_info( digitalsum, pixel_mapping, digi_sum_channel_list):
    max_digi_sum=np.max(digitalsum)
    row, col = np.unravel_index(np.argmax(digitalsum), digitalsum.shape)
    channelID=digi_sum_channel_list[row,0]
    timeID=col
    x_mean=pixel_mapping[channelID,0]
    y_mean=pixel_mapping[channelID,1]
    t_mean=col
    return def_L1_trigger_info( max_digi_sum=max_digi_sum,
                                x_mean=x_mean, y_mean=y_mean, t_mean=t_mean,
                                channelID=channelID, timeID=timeID)

def print_trigger_info(trigger_info):
    print(trigger_info.keys())
    for key in trigger_info.keys() :
        print(key," ",trigger_info[key])
    
def save_analyze_noise( data, pdf_file_out, n_samples, nsigma=12, nbinsfactor=2):
    #
    fig, ax = plt.subplots()
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_min = int(data_mean - nsigma*data_std)
    data_max = int(data_mean + nsigma*data_std)
    #data_min = 280
    #data_max = 330
    #nbins=int(2*nsigma*data_std*nbinsfactor)
    nbins=int(35*nbinsfactor)
    #nbins=11
    hist_data=ax.hist(data, bins=np.linspace(data_min, data_max, num=nbins), edgecolor='black')
    #ax.hist(data, bins=nbins)
    plt.xlabel('ADC value')
    #plt.ylabel('')
    #plt.title('npe')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    # Show the plot
    #plt.show()
    ax.get_figure().savefig(pdf_file_out)
    ax.clear();
    ax.remove();
    plt.close('all');
    #
    pdf_file_out_rates = str(str(pdf_file_out) + '_rates.pdf')
    save_and_analyze_rates_noise( hist_data, pdf_file_out_rates, n_samples)
    
def save_and_analyze_rates_noise( hist_data, pdf_file_out, n_samples):
    counts=hist_data[0]
    thresholds=hist_data[1][:-1]
    #
    print("np.sum(counts) = ",np.sum(counts))
    #
    rates = np.array([np.sum(counts[i:]) for i in np.arange(0,len(counts))])
    rates = rates/(_time_of_one_sample_s*n_samples)
    fig, ax = plt.subplots()
    ax.plot(thresholds, rates, 'bo-', linewidth=2, markersize=4)
    plt.xlabel('thresholds, ADC counts')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    # Show the plot
    #plt.show()
    ax.get_figure().savefig(pdf_file_out)
    ax.clear();
    ax.remove();
    plt.close('all');
    #
    df = pd.DataFrame({'thresholds': thresholds,
                       'counts': counts,
                       'rates': rates})
    csv_file_out=str(str(pdf_file_out)+str('.csv'))
    df.to_csv(csv_file_out, sep=' ', index=False)
    #
    return    
    
def analyze_noise( wf_noise, wf_noise_blacklist,
                   L0_digitalsum_noise, L1_digitalsum_noise,
                   L0_digitalsum_noise_blacklist, L1_digitalsum_noise_blacklist):
    #
    print(len(wf_noise))
    print(len(L0_digitalsum_noise))
    print(len(L1_digitalsum_noise))
    #
    print(wf_noise[0].shape)
    print(L0_digitalsum_noise[0].shape)
    print(L1_digitalsum_noise[0].shape)
    #
    print(_time_of_one_sample_s)
    #
    result = None
    for array in wf_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    wf_noise_arr = result
    #
    result = None
    for array in L0_digitalsum_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L0_digitalsum_noise_arr = result
    #
    result = None
    for array in L1_digitalsum_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L1_digitalsum_noise_arr = result
    #
    result = None
    for array in wf_noise_blacklist:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    wf_noise_blacklist_arr = result
    #
    result = None
    for array in L0_digitalsum_noise_blacklist:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L0_digitalsum_noise_blacklist_arr = result
    #
    result = None
    for array in L1_digitalsum_noise_blacklist:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L1_digitalsum_noise_blacklist_arr = result
    #
    print(np.mean(wf_noise_arr)," ",np.std(wf_noise_arr))
    print(np.mean(L0_digitalsum_noise_arr)," ",np.std(L0_digitalsum_noise_arr))
    print(np.mean(L1_digitalsum_noise_arr)," ",np.std(L1_digitalsum_noise_arr))
    #
    save_analyze_noise( data=wf_noise_arr.flatten(), pdf_file_out="wf_noiseNSB268MHz_arr.pdf", n_samples=len(wf_noise))
    save_analyze_noise( data=L0_digitalsum_noise_arr.flatten(), pdf_file_out="L0_digitalsum_noiseNSB268MHz_arr.pdf", n_samples=len(L0_digitalsum_noise))
    save_analyze_noise( data=L1_digitalsum_noise_arr.flatten(), pdf_file_out="L1_digitalsum_noiseNSB268MHz_arr.pdf", n_samples=len(L1_digitalsum_noise))
    #
    save_analyze_noise( data=wf_noise_blacklist_arr.flatten(), pdf_file_out="wf_noiseNSB268MHz_blacklist_arr.pdf", n_samples=len(wf_noise_blacklist))
    save_analyze_noise( data=L0_digitalsum_noise_blacklist_arr.flatten(), pdf_file_out="L0_digitalsum_noiseNSB268MHz_blacklist_arr.pdf", n_samples=len(L0_digitalsum_noise_blacklist))
    save_analyze_noise( data=L1_digitalsum_noise_blacklist_arr.flatten(), pdf_file_out="L1_digitalsum_noiseNSB268MHz_blacklist_arr.pdf", n_samples=len(L1_digitalsum_noise_blacklist))
    #
    return
    
def evtloop_noise(datafilein, nevmax, pixel_mapping, L0_trigger_pixel_cluster_list, L1_trigger_DBSCAN_pixel_cluster_list):
    #
    print("evtloop_noise")
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    event_info_list=[]
    clusters_info_list=[]
    #
    wf_trigger_pixel_list=np.array([i for i in np.arange(0,pixel_mapping.shape[0])])
    wf_trigger_pixel_list=np.reshape(wf_trigger_pixel_list,(pixel_mapping.shape[0],1))
    #
    wf_noise_list=[]
    wf_noise_blacklist_list=[]
    L0_digitalsum_noise_list=[]
    L0_digitalsum_noise_blacklist_list=[]
    L1_digitalsum_noise_list=[]
    L1_digitalsum_noise_blacklist_list=[]
    #
    for ev in sf:
        #
        LSTID=ev['telescope_events'].keys()
        #
        wf_list=[]
        n_pe_per_tel_list=[]
        LSTID_list=[]
        for i in LSTID :
            wf_list.append(ev['telescope_events'][i]['adc_samples'][0])
            n_pe_per_tel_list.append(int(ev['photoelectrons'][i-1]['n_pe']))
            LSTID_list.append(int(i-1))
        #
        #
        L0_digitalsum_list=[]
        for wf, npe, lst_id in zip( wf_list, n_pe_per_tel_list, LSTID_list) :
            try:
                if npe == _npe_noise:
                    if(len(wf_noise_list)<_n_max_noise_events):
                        channels_blacklist=np.unique(np.reshape(np.array(ev['photoelectrons'][lst_id]['pixel_id']),(npe,1)))
                        wf_noise_list.append(wf)
                        wf_noise_blacklist_list.append(digital_w_sum(wf=wf, digi_sum_channel_list=wf_trigger_pixel_list, channels_blacklist=channels_blacklist))
                        L0_digitalsum_noise_list.append(digital_sum(wf=wf, digi_sum_channel_list=L0_trigger_pixel_cluster_list))
                        L0_digitalsum_noise_blacklist_list.append(digital_w_sum(wf=wf, digi_sum_channel_list=L0_trigger_pixel_cluster_list, channels_blacklist=channels_blacklist))
                        L1_digitalsum_noise_blacklist_list.append(digital_w_sum(wf=wf, digi_sum_channel_list=L1_trigger_DBSCAN_pixel_cluster_list, channels_blacklist=channels_blacklist))
                        L1_digitalsum_noise_list.append(digital_sum(wf=wf, digi_sum_channel_list=L1_trigger_DBSCAN_pixel_cluster_list))
            except:
                pass
        #
        ev_counter=ev_counter+1
        #print_ev(ev)
        if (ev_counter >= nevmax and nevmax > 0):
            break
        if (it_cout%_event_modulo==0) :
            toc = time.time()
            print('{:10d} {:10.2f} s'.format(it_cout, toc - tic))
            tic = time.time()
        it_cout = it_cout + 1
        
    sf.close()
    #
    print("L0_digitalsum_noise_list                    ",len(L0_digitalsum_noise_list))
    print("L0_digitalsum_noise_blacklist_list          ",len(L0_digitalsum_noise_blacklist_list))
    print("L0_digitalsum_noise_list[0].shape           ",L0_digitalsum_noise_list[0].shape)
    print("L0_digitalsum_noise_blacklist_list[0].shape ",L0_digitalsum_noise_blacklist_list[0].shape)
    #
    analyze_noise( wf_noise=wf_noise_list,
                   wf_noise_blacklist=wf_noise_blacklist_list,
                   L0_digitalsum_noise=L0_digitalsum_noise_list,
                   L1_digitalsum_noise=L1_digitalsum_noise_list,
                   L0_digitalsum_noise_blacklist=L0_digitalsum_noise_blacklist_list,
                   L1_digitalsum_noise_blacklist=L1_digitalsum_noise_blacklist_list)
    #        
    return

def evtloop(datafilein, nevmax, pixel_mapping, L1_trigger_pixel_cluster_list, L3_trigger_DBSCAN_pixel_cluster_list):
    #
    print("evtloop")
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    digital_sum_threshold=[]
    thresholds=np.linspace(6000, 7000, num=1000)
    #
    event_info_list=[]
    #
    pixel_mapping_extended=extend_pixel_mapping( pixel_mapping=pixel_mapping, channel_list=L3_trigger_DBSCAN_pixel_cluster_list, number_of_wf_time_samples=_n_of_time_sample)
    #
    for ev in sf:
        #
        LSTID=ev['telescope_events'].keys()
        #
        #
        wf_list=[]
        n_pe_per_tel_list=[]
        LSTID_list=[]
        L1_trigger_info_list=[]
        DBSCAN_clusters_info_list=[]
        for i in LSTID :
            wf_list.append(ev['telescope_events'][i]['adc_samples'][0])
            n_pe_per_tel_list.append(int(ev['photoelectrons'][i-1]['n_pe']))
            LSTID_list.append(int(i-1))
        #
        #
        for wf, npe, lst_id in zip( wf_list, n_pe_per_tel_list, LSTID_list) :
            try:                
                L1_digitalsum = digital_sum(wf=wf, digi_sum_channel_list=L1_trigger_pixel_cluster_list)
                L3_digitalsum = digital_sum(wf=wf, digi_sum_channel_list=L3_trigger_DBSCAN_pixel_cluster_list)
                L1_trigger_info = get_L1_trigger_info(digitalsum=L1_digitalsum, pixel_mapping=pixel_mapping, digi_sum_channel_list=L1_trigger_pixel_cluster_list)
                #
                DBSCAN_clusters_info = get_DBSCAN_clusters( digitalsum = L3_digitalsum,
                                                            pixel_mapping = pixel_mapping,
                                                            pixel_mapping_extended = pixel_mapping_extended,
                                                            channel_list=L3_trigger_DBSCAN_pixel_cluster_list,
                                                            time_norm = _time_norm,
                                                            digitalsum_threshold = _DBSCAN_digitalsum_threshold,
                                                            DBSCAN_eps = _DBSCAN_eps,
                                                            DBSCAN_min_samples = _DBSCAN_min_samples)
                #print_trigger_info(trigger_info=DBSCAN_clusters_info)
            except:
                L1_trigger_info=def_L1_trigger_info()
                DBSCAN_clusters_info = def_clusters_info()
            #
            #
            L1_trigger_info_list.append(L1_trigger_info)
            DBSCAN_clusters_info_list.append(DBSCAN_clusters_info)
        #
        #
        
        #clusters_info['event_ID'] = int(ev['event_id'])
        #clusters_info_list.append(clusters_info)
        #
        #event_info_list.append([ev['event_id'],
        #                        ev['mc_shower']['energy'],
        #                        ev['mc_shower']['azimuth'],
        #                        ev['mc_shower']['altitude'],
        #                        ev['mc_shower']['h_first_int'],
        #                        ev['mc_shower']['xmax'],
        #                        ev['mc_shower']['hmax'],
        #                        ev['mc_shower']['emax'],
        #                        ev['mc_shower']['cmax'],
        #                        ev['mc_event']['xcore'],
        #                        ev['mc_event']['ycore'],
        #                        ev['telescope_events'][1]['header']['readout_time'],
        #                        len(ev['photons'][0]),
        #                        ev['photoelectrons'][0]['n_pe'],
        #                        (ev['photoelectrons'][0]['n_pixels']-np.sum(ev['photoelectrons'][0]['photoelectrons']==0)),
        #                        clusters_info['event_ID'],
        #                        clusters_info['n_digitalsum_points'],
        #                        clusters_info['n_clusters'],
        #                        clusters_info['n_points'],
        #                        clusters_info['x_mean'],
        #                        clusters_info['y_mean'],
        #                        clusters_info['t_mean'],
        #                        clusters_info['channelID'],
        #                        clusters_info['timeID']])                               
        # 
        #
        #print(clusters_info)
        #
        ev_counter=ev_counter+1
        #
        #
        if (ev_counter >= nevmax and nevmax > 0):
            break
        if (it_cout%_event_modulo==0) :
            toc = time.time()
            print('{:10d} {:10.2f} s'.format(it_cout, toc - tic))
            tic = time.time()
        it_cout = it_cout + 1
        
    sf.close()
    
    return  event_info_list

    
def main():
    pass
    
if __name__ == "__main__":
    if (len(sys.argv)==8 and (str(sys.argv[1]) == "--noise")):
        #
        simtelIn = str(sys.argv[2])
        headeroutpkl = str(sys.argv[3])
        headeroutcsv = str(sys.argv[4])
        pixel_mapping_csv = str(sys.argv[5])
        isolated_flower_seed_super_flower_csv = str(sys.argv[6])
        isolated_flower_seed_flower_csv = str(sys.argv[7])
        #
        print("sys.argv[1]                           = ", sys.argv[1])
        print("simtelIn                              = ", simtelIn)
        print("headeroutpkl                          = ", headeroutpkl)
        print("headeroutcsv                          = ", headeroutcsv)
        print("pixel_mapping_csv                     = ", pixel_mapping_csv)
        print("isolated_flower_seed_super_flower_csv = ", isolated_flower_seed_super_flower_csv)
        print("isolated_flower_seed_flower_csv       = ", isolated_flower_seed_flower_csv)
        #
        print_setup()
        #
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        isolated_flower_seed_flower = np.genfromtxt(isolated_flower_seed_flower_csv,dtype=int) 
        isolated_flower_seed_super_flower = np.genfromtxt(isolated_flower_seed_super_flower_csv,dtype=int)
        #
        evtloop_noise( datafilein=simtelIn, nevmax=-1,
                       pixel_mapping=pixel_mapping,
                       L0_trigger_pixel_cluster_list=isolated_flower_seed_super_flower,
                       L1_trigger_DBSCAN_pixel_cluster_list=isolated_flower_seed_flower)
        #
    elif (len(sys.argv)==8 and (str(sys.argv[1]) == "--trg")):
        #
        simtelIn = str(sys.argv[2])
        headeroutpkl = str(sys.argv[3])
        headeroutcsv = str(sys.argv[4])
        pixel_mapping_csv = str(sys.argv[5])
        isolated_flower_seed_super_flower_csv = str(sys.argv[6])
        isolated_flower_seed_flower_csv = str(sys.argv[7])
        #
        print("sys.argv[1]                           = ", sys.argv[1])
        print("simtelIn                              = ", simtelIn)
        print("headeroutpkl                          = ", headeroutpkl)
        print("headeroutcsv                          = ", headeroutcsv)
        print("pixel_mapping_csv                     = ", pixel_mapping_csv)
        print("isolated_flower_seed_super_flower_csv = ", isolated_flower_seed_super_flower_csv)
        print("isolated_flower_seed_flower_csv       = ", isolated_flower_seed_flower_csv)
        #
        print_setup()
        #
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        isolated_flower_seed_flower = np.genfromtxt(isolated_flower_seed_flower_csv,dtype=int) 
        isolated_flower_seed_super_flower = np.genfromtxt(isolated_flower_seed_super_flower_csv,dtype=int)
        #
        evtloop( datafilein=simtelIn, nevmax=-1,
                 pixel_mapping=pixel_mapping,
                 L1_trigger_pixel_cluster_list=isolated_flower_seed_super_flower,
                 L3_trigger_DBSCAN_pixel_cluster_list=isolated_flower_seed_flower)
        #
    else:
        print(" --> HELP info")
        print("len(sys.argv) = ",len(sys.argv))
        print("sys.argv      = ",sys.argv)
        print(" ---> for noise")
        print(" [1] --noise")
        print(" [2] simtelIn")
        print(" [3] headeroutpkl")
        print(" [4] headeroutcsv")
        print(" [5] pixel_mapping_csv")
        print(" [6] isolated_flower_seed_super_flower_csv")
        print(" [7] isolated_flower_seed_flower_csv")
        print(" ---> for noise")
        print(" [1] --trg")
        print(" [2] simtelIn")
        print(" [3] headeroutpkl")
        print(" [4] headeroutcsv")
        print(" [5] pixel_mapping_csv")
        print(" [6] isolated_flower_seed_super_flower_csv")
        print(" [7] isolated_flower_seed_flower_csv")