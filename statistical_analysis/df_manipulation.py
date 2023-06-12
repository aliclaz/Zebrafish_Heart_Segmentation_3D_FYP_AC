import pandas as pd
from imgPreprocessing import get_hpf
from .maths_stats import get_class_volumes, get_class_volume_msd, add_vols_msd, get_CIs, stat_sig_diff_test

__all__ = ['gm_df_calcs', 'healthy_df_calcs']

def gm_df_calcs(masks, classes, hpf, gm, scales, in_path, out_path):

    # Get one of the three default hpf values based on which is closest to to the entered hpf

    mod_hpf = get_hpf(hpf)

    # If only one image was entered, calculate its volume

    if len(masks) == 1:
        observed = get_class_volumes(masks[0], scales[0])

    # If multiple images entered, calculate the mean and standard deviation of the volume of each class

    else:
        observed = get_class_volume_msd(masks, classes, scales, mod_hpf, gm)

    # Load dataframes containing healthy volumes or healthy volumes and analysis of images previously entered

    stats = pd.read_csv(in_path+'{}HPF_stat_results.csv'.format(mod_hpf))

    # Put the healthy means and confidence intervals from the dataframe into lists

    healthy_means = stats['Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(mod_hpf)].tolist()
    healthy_CIs = stats['Confidence Interval of Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(mod_hpf)].tolist()

    # If 1 image was entered, determine whether volume of each class is statistically significantly different from the mean healthy
    # volume of each class

    results = []

    if len(masks) == 1:
        for i in range(len(classes)):
            result = stat_sig_diff_test(observed[i], healthy_means[i], healthy_CIs[i])
            results.append(result)

    # If multiple images were entered, determine whether the mean volume of each class is statistically significantly different from
    # the mean healthy volume of each class

    else:
        sds = observed['Standard Deviation of {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)].tolist()
        CIs = get_CIs(sds, len(masks), classes, hpf, gm)
        means = observed['Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)].tolist()

        for i in range(len(classes)):
            if means[i] < healthy_means[i]:
                result = stat_sig_diff_test(means[i] + CIs[i], healthy_means[i], healthy_CIs[i])
            elif means[i] > healthy_means[i]:
                result = stat_sig_diff_test(means[i] - CIs[i], healthy_means[i], healthy_CIs[i])
            else:
                result = False
            results.append(result)

    # Add the results to a dataframe with a labelled columns and rows

    column = ['Statistically Significant Difference Between {} Volume and Healthy Volume at {}'. format(gm, hpf)]
    results_df = pd.DataFrame(results, index=classes, columns=column, dtype=bool)

    # Add the results to the originial dataframe

    if len(masks) == 1:
        column = ['{} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)]
        observed = pd.DataFrame(observed, index=classes, columns=column, dtype=float)
        stats = stats.append(observed)
    else:
        stats = stats.append(observed)
    stats = stats.append(results_df)

    # Remove unwanted classes from the dataframe for display

    stats_disp = stats.drop(['Background', 'Endocardium', 'Noise'])
    print(stats_disp)

    # Save the regular results (for future analysis) in the input path so they can be accessed again and the display results in the
    # output path so the user can access them - as CSV files

    stats.to_csv(in_path+'{}HPF_stat_analysis.csv'.format(hpf))
    stats_disp.to_csv(out_path+'{}HPF_stat_display.csv'.format(hpf))

    # Load the dataframe containing the number of samples used for calulating the results at the entered hpf and gm

    n_samples = pd.read_csv(in_path+'n_samples.csv')

    # Add the number of samples to the value in an existing shell or create a new cell with a new row and/or column and put the number
    # of samples into it

    n_new = pd.DataFrame([len(masks)], index=[hpf], columns=[gm], dtype=int)
    n_samples = n_samples.append(n_new)

    # Save the dataframe

    n_samples.to_csv(in_path+'n_samples.csv')

def healthy_df_calcs(masks, classes, scales, hpf, out_path):
    
    # If the calculations for healthy volumes are being completed for the first time, the number of original samples will be 0
    # Calculate the mean and standard deviations of the volumes of the masks for each class
    
    msd_class_vols_df = get_class_volume_msd(masks, classes, scales, hpf, 'Healthy')

    # Use the means and standard deviations of each class to calculate the confidence intervals of the healthy volumes

    sds = msd_class_vols_df['Standard Deviation of Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(hpf)].tolist()
    class_vol_CIs_df = get_CIs(sds, len(masks), classes, hpf, 'Healthy')

    # Add all information into one dataframe, display it and save it as a CSV file
    # which can be viewed outside of the code or terminal

    # Create display dataframe and analysis dataframe as all classes needed for analysis of new images

    stats = pd.DataFrame()
    stats = stats.append(msd_class_vols_df)
    stats = stats.append(class_vol_CIs_df)

    # Remove unwanted classes for displaying the results

    stats_disp = stats.drop(['Endocardium', 'Background', 'Noise'])
    print(stats_disp)

    # Save the regular results (for future analysis) in the input path so they can be accessed again and the display results in the
    # output path so the user can access them - as CSV files

    stats.to_csv(out_path+'{}HPF_stat_analysis.csv'.format(hpf))
    stats_disp.to_csv(out_path+'{}HPF_stat_results.csv'.format(hpf))

    # Create a dataframe of the number of samples for each gm at each stage of development and save it as a csv file

    n_samples = pd.DataFrame(index=[30, 36, 48], columns=['Healthy'], dtype=int)
    n_samples[hpf, 'healthy'] = len(masks)

    n_samples.to_csv(out_path+'n_samples.csv')

def add_df_calcs(new_masks, classes, hpf, gm, scales, in_path, out_path):

    # Get one of the three default hpf values based on which is closest to to the entered hpf

    mod_hpf = get_hpf(hpf)

    # Load dataframe containing all sample numbers

    n_samples = pd.read_csv(in_path+'n_samples.csv')

    # Get the number of samples for the specific hpf and gm entered

    n = n_samples[hpf, gm]

    # Load dataframe containing healthy mean volumes and analysis of images previously entered

    stats = pd.read_csv(in_path+'{}HPF_stat_results.csv'.format(mod_hpf))

    # Put the volume or mean volume of each class for the specific hpf and gm entered into a list if 1 image was originally entered or if
    # multiple images were originally entered respectively

    if n == 1:
        og = stats['{} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)]

    else:
        og = stats['Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)].tolist()

    # Calculate the new mean volume of the specific hpf and gm entered

    new_class_volume_msd = add_vols_msd(n, og, new_masks, classes, scales, hpf, gm)
    
    new_sds = new_class_volume_msd['Standard Deviation of {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)].tolist()
    new_means = new_class_volume_msd['Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)].tolist()

    # Put the healthy means and confidence intervals from the dataframe into lists

    healthy_means = stats['Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(mod_hpf)].tolist()
    healthy_CIs = stats['Confidence Interval of Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(mod_hpf)].tolist()

    # Calculate the new confidence intervals and test for significant difference with changes

    new_CIs = get_CIs(new_sds, n, classes, hpf, gm)

    new_results = []

    for i in range(len(classes)):
        if new_means[i] < healthy_means[i]:
            result = stat_sig_diff_test(new_means[i] + new_CIs[i], healthy_means[i], healthy_CIs[i])
        elif new_means[i] > healthy_means[i]:
            result = stat_sig_diff_test(new_means[i] - new_CIs[i], healthy_means[i], healthy_CIs[i])
        else:
            result = False
        new_results.append(result)

    # Replace old volumes with new mean volumes and add new standard deviations, confidence intervals and result of statistical
    # difference test when only 1 image was originally added

    if n == 1:
        stats = stats.rename({'{} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf): 'Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)})
        stats['Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)] = new_means
        
        columns = list(stats.columns)
        mean_col_loc = columns.index('Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf))
        stats = stats.insert(mean_col_loc + 1, 'Standard Deviation of {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf), new_sds)
        
        sds_col_loc = columns.index('Standard Deviation of {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf))
        stats.insert(sds_col_loc + 1, 'Confidence Interval of Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf), new_CIs)
    else:
        stats['Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)] = new_means

        stats['Standard Deviation of {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)] = new_sds

        stats['Confidence Interval of Mean {} Volume (\u03bcm\u00b2) at {}HPF'.format(gm, hpf)] = new_CIs

    stats['Statistically Significant Difference Between {} Volume and Healthy Volume at {}'.format(gm, hpf)] = new_results

    # Remove unwanted classes for displaying the results

    stats_disp = stats.drop(['Endocardium', 'Background', 'Noise'])
    print(stats_disp)

    # Save the regular results (for future analysis) in the input path so they can be accessed again and the display results in the
    # output path so the user can access them - as CSV files

    stats.to_csv(out_path+'{}HPF_stat_analysis.csv'.format(out_path))
    stats_disp.to_csv(out_path+'{}HPF_stat_results.csv'.format(out_path))

    # Add the number of original samples to the number of new samples to the sample size dataframe and save it as a CSV

    n_samples[hpf, gm] = n + len(new_masks)

    n_samples.to_csv(out_path+'n_samples.csv')

def add_healthy_df_calcs(new_healthy_masks, classes, hpf, scales, in_path, out_path):

    # Load dataframe containing all sample numbers

    n_samples = pd.read_csv(in_path+'n_samples.csv')

    # Get the number of samples for the specific hpf and gm entered

    n = n_samples[hpf, 'Healthy']

    # Load dataframe containing healthy mean volumes and analysis of images previously entered

    stats = pd.read_csv(in_path+'{}HPF_stat_results.csv'.format(hpf))

    # Put the original healthy mean class volumes into a list

    og_healthy_means = stats['Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(hpf)].tolist()

    # Calculate the new healthy mean and standard deviations of each class volume

    new_healthy_msd = add_vols_msd(n, og_healthy_means, new_healthy_masks, classes, scales, hpf, 'Healthy')

    new_healthy_sds = new_healthy_msd['Standard Deviation of Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(hpf)].tolist()
    new_healthy_means = new_healthy_msd['Mean Healthy Volume (\u03bcm\u00b2) at {}HPF'.format(hpf)].tolist()

    # Calculate the new confidence intervals

    new_healthy_CIs = get_CIs(new_healthy_sds, n, classes, hpf, 'Healthy')

    # Create lists of all the gm and hpf values that have already been used

    all_hpf = list(n_samples.index)
    all_gm = list(n_samples.columns)

    for i in all_hpf:
        for j in all_gm:
            if n_samples[all_hpf[i], all_gm[j]].isnull().values.any:
                del all_hpf[i]
                del all_gm[i]
    list(map(str, all_hpf))
    list(map(str, all_gm))

    columns = list(stats.columns)

    all_gm.remove('Healthy')

    # Check all the possible column titles if they contain all possible combinations of hpf and gm except healthy
    # Then check if it is the mean or confidence interval column for that combination, if it is test for significicant difference
    # from healthy mean volumes put the results for each class into a list and put these lists for each combination into another list

    new_results = []

    for i in range(len(all_gm)):
        for j in range(len(all_hpf)):
            for k in range(len(columns)):
                if (all_gm[i], all_hpf[j] in columns[k]) and (n_samples[list(map(int, all_hpf)), list(map(int, all_gm))] == 1):
                    if ('Volume' in columns[k]) and ('Mean', 'Confidence', 'Standard', 'Difference' not in columns[k]):
                        vols = stats[columns[k]].tolist()

                    if len(vols) != 0:
                        for i in range(len(classes)):
                            result = stat_sig_diff_test(vols[l], new_healthy_means[l], new_healthy_CIs[l])
                            new_results.append(result)
                        stats['Statistically Significant Difference Between {} Volume and Healthy Volume at {}'.format(all_gm[i], all_hpf[j])] = new_results
                    
                elif (all_gm[i] in columns[k]) and (all_hpf[j] in columns[k]) and (n_samples[list(map(int, all_hpf)), 
                                                                                             list(map(int, all_gm))] > 1):
                    if ('Mean' in columns[k]) and ('Confidence' not in columns[k+4]):
                        means = stats[columns[k]].tolist()
                    elif 'Confidence' in columns[k]:
                        CIs = stats[columns[k]].tolist()

                    if len(vols) != 0:
                        for l in range(len(classes)):
                            if means[l] < new_healthy_means[l]:
                                result = stat_sig_diff_test(means[l] + CIs[l], new_healthy_means[l], new_healthy_CIs[l])
                            elif means[l] > new_healthy_means[l]:
                                result = stat_sig_diff_test(means[l] - CIs[l], new_healthy_means[l], new_healthy_CIs[l])
                            else:
                                result = False
                            new_results.append(result)
                        stats['Statistically Significant Difference Between {} Volume and Healthy Volume at {}'.format(all_gm[i], all_hpf[j])] = new_results
    
    # Remove unwanted classes for display purposes

    stats_disp = stats.drop(['Background', 'Endocardium', 'Noise'])
    print(stats_disp)

    # Save the regular results (for future analysis) in the input path so they can be accessed again and the display results in the
    # output path so the user can access them - as CSV files

    stats.to_csv(out_path+'{}HPF_stat_analysis.csv'.format(out_path))
    stats_disp.to_csv(out_path+'{}HPF_stat_results.csv'.format(out_path))

    # Add the number of original samples to the number of new samples to the sample size dataframe and save it as a CSV

    n_samples[hpf, 'Healthy'] = n + len(new_healthy_masks)

    n_samples.to_csv(out_path+'n_samples.csv')