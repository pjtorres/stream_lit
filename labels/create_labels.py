"""
Created on 07-05-2023
@author: Pedro J. Torres
"""

import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import tkinter as tk
import csv
from tkinter import ttk
from tkinter.messagebox import showinfo
import itertools




# Analysis types
# Did not add 'CRCi Fecal' as it seems to need additonal files I do not have. Wll leave this up to Hiro
analysis_options = ['Argonaut Blood for Crown', 'Argonaut Fecal', 'Argonaut Tissue-CRC',
                    'Argonaut Tissue-low-high_risk','Baby Fecal-FMT',
                    'Baby Fecal','Blood-draw tubes','COH Fecal','poo kit labels']


# Numerical start and end values
start_value = 0
end_value = 0

st.title("Label Maker")

st.markdown("""

Generating bulk import spreadsheets to use with the dymo printers

## Argonaut Fecal
Labels to use in the lab for fecal sample processing

## Argonaut Tissue
Tissue labels to send to the CRO's

## Argonaut Blood for Crown
Blood labels for Crown Bio

## Poo Kit Labels
Labels for our in-house poo kits

""")

# Dropdown for labeling type
analysis_type = st.selectbox("Select Labeling Type", analysis_options)

# Numeric input for start and end values
start_value = st.number_input("Enter Sample Start Value", value=start_value)
end_value = st.number_input("Enter Sample End Value", value=end_value)

# Run analysis button
if st.button("Run Analysis"):
    # Perform analysis based on the selected analysis type
    if analysis_type == 'Argonaut Blood for Crown':
        # Code for Analysis 1
        st.write("Running Argonaut Blood for Crown...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end = end_value

        #Edit the number of labels for each sample type
        pbmc_jj_num = 2
        pbmc_num = 2
        plasma_jj_num = 4
        plasma_num = 2
        paxdna_jj_num = 1
        paxrna_jj_num = 1
        serum_jj_num = 4
        streck_jj_num = 4


        num_tot =  sum([pbmc_jj_num, pbmc_num, plasma_jj_num, plasma_num, paxdna_jj_num, paxrna_jj_num, serum_jj_num,
             streck_jj_num])

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(start, tru_end)))
        count = tru_end - start


        #Create a list of values for the number of each sample
        nums = [pbmc_jj_num, pbmc_num, plasma_jj_num, plasma_num, paxdna_jj_num, paxrna_jj_num, serum_jj_num,
             streck_jj_num]

        #list of sample types
        sample_types = ['PBMC', 'PBMC', 'Plasma', 'Plasma', 'PAXDNA', 'PAXRNA', 'Serum', 'Streck']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')

        # list of owners in the correct orientation for samples
        owner_types = ['JJ-92108', 'PB', 'JJ-92108', 'PB', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108']

        #create list of owners
        owner_tot = []
        for x in range(0, count):
            for sample in zip(owner_types, nums):
                for i in range(sample[1]):
                    owner_tot.append(f'{sample[0]}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2', 'Line3']

        with open('Crown Blood Labels.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv("Crown Blood Labels.csv").sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2' , 'Line3' : 'Line3_2'})

        label_sheet = df2.join(df1, how='right')
        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} Crown Blood Labels 2up.csv'
        #label_sheet.to_csv(f'{lab_pbt[0] + " to " + lab_pbt[-1]} Crown Blood Labels 2up.csv', index=False)

        # Generate DataFrame
        # label_sheet = pd.DataFrame({'Label': ['Label 1', 'Label 2', 'Label 3']})

        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Argonaut Blood for Crown CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Argonaut Fecal':
        # Code for Analysis 2
        st.write("Running Argonaut Fecal...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end = end_value

        #Edit the number of labels for each sample type
        pbs_num = 0
        pbs_jj_num = 0
        bank_num = 0
        bank_jj_num = 0
        rna_num = 0
        rna_jj_num = 0
        fmt_num = 0
        fmt_jj_num = 0
        stool_jj_num = 1

        num_tot =  sum([pbs_num, pbs_jj_num, bank_num, bank_jj_num, rna_num, rna_jj_num, fmt_num,
             fmt_jj_num, stool_jj_num])

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(start, tru_end)))
        count = tru_end - start


        #Create a list of values for the number of each sample
        nums = [pbs_num, pbs_jj_num, bank_num, bank_jj_num, rna_num, rna_jj_num, fmt_num,
             fmt_jj_num, stool_jj_num]

        #list of sample types
        sample_types = ['PBS', 'PBS', 'Bank', 'Bank', 'RNA', 'RNA', 'FMT', 'FMT', 'Stool']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')

        # list of owners in the correct orientation for samples
        owner_types = ['PB', 'JJ-92108', 'PB', 'JJ-92108', 'PB', 'JJ-92108', 'PB', 'JJ-92108', 'JJ-92108']

        #create list of owners
        owner_tot = []
        for x in range(0, count):
            for sample in zip(owner_types, nums):
                for i in range(sample[1]):
                    owner_tot.append(f'{sample[0]}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2', 'Line3']

        with open('Stool Label.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv("Stool Label.csv").sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2' , 'Line3' : 'Line3_2'})

        label_sheet = df2.join(df1, how='right')
        filename=f'{lab_pbt[0] + " to " + lab_pbt[-1]} NEAT stool label 2up.csv'

        # label_sheet.to_csv(f'{lab_pbt[0] + " to " + lab_pbt[-1]} NEAT stool label 2up.csv', index=False)



        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename} CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Argonaut Tissue-CRC':
        # Code for Analysis 3
        st.write("Running Argonaut Tissue-CRC...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        #Enter Start and End PBT numbers (numbers only)
        site = "09"


        patient_start = start_value
        patient_end = end_value

        #Edit the number of labels for each sample type
        L_for_num = 1
        A_for_num = 2
        D_for_num = 2
        S_for_num = 3
        L_pax_num = 1
        A_pax_num = 2
        D_pax_num = 2
        S_pax_num = 3



        num_tot =  sum([L_for_num, A_for_num, D_for_num, S_for_num, L_pax_num, A_pax_num, D_pax_num, S_pax_num])

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (patient_end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(patient_start, tru_end)))
        count = tru_end - patient_start


        #Create a list of values for the number of each sample
        nums = [L_for_num, A_for_num, D_for_num, S_for_num, L_pax_num, A_pax_num, D_pax_num, S_pax_num]

        #list of sample types
        sample_types = ['L-For', 'A-For', 'D-For', 'S-For', 'L-PAX', 'A-PAX', 'D-PAX', 'S-PAX']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')

        # list of owners in the correct orientation for samples
        owner_types = ['JJ-92108', 'JJ-92108', 'JJ-92108','JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108','JJ-92108']

        #create list of owners
        owner_tot = []
        for x in range(0, count):
            for sample in zip(owner_types, nums):
                for i in range(sample[1]):
                    owner_tot.append(f'{sample[0]}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:03d}".format
        num_tot_list = list(map(formatter, num_tot_list))


        lab_pbt = []
        for value in num_tot_list:
            pbt = site + '-' + f'{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2', 'Line3']

        with open(f'{lab_pbt[0] + " to " + lab_pbt[-1]} CRC - Tissue 2up.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv(f'{lab_pbt[0] + " to " + lab_pbt[-1]} CRC - Tissue 2up.csv').sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2' , 'Line3' : 'Line3_2'})

        label_sheet = df1.join(df2, how='right')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} CRC - Tissue 2up.csv'



        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Argonaut Tissue-CRC CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Argonaut Tissue-low-high_risk':
        # Code for Analysis 3
        st.write("Running Argonaut Tissue-low-high_risk...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        #Enter Start and End PBT numbers (numbers only)
        site = "14"


        patient_start = start_value
        patient_end = end_value

        #Edit the number of labels for each sample type
        c_for_num = 1
        ac_for_num = 1
        tc_for_num = 1
        dc_for_num = 1
        sc_for_num = 1
        r_for_num = 1
        c_pax_num = 1
        ac_pax_num = 1
        tc_pax_num = 1
        dc_pax_num = 1
        sc_pax_num = 1
        r_pax_num = 1
        b_nl_num = 0
        arc_num = 0



        num_tot =  sum([c_for_num, ac_for_num, tc_for_num, dc_for_num, sc_for_num, r_for_num,
                        c_pax_num, ac_pax_num, tc_pax_num, dc_pax_num, sc_pax_num, r_pax_num, b_nl_num, arc_num])

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (patient_end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(patient_start, tru_end)))
        count = tru_end - patient_start


        #Create a list of values for the number of each sample
        nums = [c_for_num, ac_for_num, tc_for_num, dc_for_num, sc_for_num, r_for_num,
                        c_pax_num, ac_pax_num, tc_pax_num, dc_pax_num, sc_pax_num, r_pax_num, b_nl_num, arc_num]

        #list of sample types
        sample_types = ['C-For', 'AC-For', 'TC-For', 'DC-For', 'SC-For', 'R-For', 'C-PAX', 'AC-PAX', 'TC-PAX', 'DC-PAX', 'SC-PAX', 'R-PAX', 'B-NL', 'Arc']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')

        # list of owners in the correct orientation for samples
        owner_types = ['JJ-92108', 'JJ-92108', 'JJ-92108','JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108', 'JJ-92108',
                      'JJ-92108', 'JJ-92108']

        #create list of owners
        owner_tot = []
        for x in range(0, count):
            for sample in zip(owner_types, nums):
                for i in range(sample[1]):
                    owner_tot.append(f'{sample[0]}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:03d}".format
        num_tot_list = list(map(formatter, num_tot_list))


        lab_pbt = []
        for value in num_tot_list:
            pbt = site + '-' + f'{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2', 'Line3']

        with open(f'{lab_pbt[0] + " to " + lab_pbt[-1]} Tissue 2up.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv(f'{lab_pbt[0] + " to " + lab_pbt[-1]} Tissue 2up.csv').sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2' , 'Line3' : 'Line3_2'})

        label_sheet = df1.join(df2, how='right')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} Tissue 2up.csv'



        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Argonaut Tissue-low-high_risk CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Baby Fecal-FMT':
        # Code for Analysis 3
        st.write("Running Baby Fecal-FMT...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        # Generate DataFrame
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end_value = end_value
        number_to_add = 200

        end = start + number_to_add - 1

        #Edit the number of labels for each sample type
        stool_cup = 0
        pbs_num = 0
        bank_num = 0
        rna_num = 0
        fmt_num = 12
        pH_num = 0

        num_tot =  sum([stool_cup, pbs_num, bank_num, fmt_num, rna_num, pH_num])


        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(start, tru_end)))
        count = tru_end - start


        #Create a list of values for the number of each sample
        nums = [stool_cup, pbs_num, bank_num, fmt_num, rna_num, pH_num]

        #list of sample types
        sample_types = ['Stool Cup', 'PBS', 'Bank', 'FMT', 'RNA', 'pH']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        #print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2']

        with open('Stool Label.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv("Stool Label.csv").sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2'})

        label_sheet = df2.join(df1, how='right')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} stool FMT 2up.csv'


        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Baby Fecal-FMT CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Baby Fecal':
        # Code for Analysis 3
        st.write("Running Baby Fecal...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        # Generate DataFrame
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end_value=end_value
        number_to_add = 63

        end = start + number_to_add - 1

        #Edit the number of labels for each sample type
        stool_cup = 0
        pbs_num = 6
        bank_num = 0
        rna_num = 0
        fmt_num = 12
        pH_num = 1

        num_tot =  sum([stool_cup, pbs_num, bank_num, fmt_num, rna_num, pH_num])


        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(start, tru_end)))
        count = tru_end - start


        #Create a list of values for the number of each sample
        nums = [stool_cup, pbs_num, bank_num, fmt_num, rna_num, pH_num]

        #list of sample types
        sample_types = ['Stool Cup', 'PBS', 'Bank', 'FMT', 'RNA', 'pH']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        #print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2']

        with open('Stool Label.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv("Stool Label.csv").sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2'})

        label_sheet = df2.join(df1, how='right')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} baby stool label 2up.csv'

        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Baby Fecal CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'Blood-draw tubes':
        # Code for Analysis 3
        st.write("Running Blood-draw tubes...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        # Generate DataFrame
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end = end_value
        copies = 14

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, copies) for x in range(start, tru_end)))

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        count = tru_end - start

        # adding 'PBT' to lab_pbt label
        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)

        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.DataFrame(lab_pbt, columns = ['PBT_Top']).sort_values(by='PBT_Top', ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        #Rename df2 columns for the subsequent join command
        df2 = df2.rename(columns={ 'PBT_Top' : 'PBT_Bottom'})

        #Join df2 to df1. If there's an odd number of samples the "how" flag is important
        label_sheet = df1.join(df2, how='left')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} blood draw label-2up.csv'

        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Blood-draw tubes CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'COH Fecal':
        # Code for Analysis 3
        st.write("Running COH Fecal...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        # Generate DataFrame
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end = end_value

        #Edit the number of labels for each sample type
        pbs_num = 5
        pbs_jj_num = 0
        bank_num = 2
        bank_jj_num = 0
        rna_num = 2
        rna_jj_num = 0
        fmt_num = 14
        fmt_jj_num = 0
        stool_jj_num = 0
        pH_num = 1

        num_tot =  sum([pbs_num, pbs_jj_num, bank_num, bank_jj_num, rna_num, rna_jj_num, fmt_num,
             fmt_jj_num, stool_jj_num, pH_num])

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_tot) for x in range(start, tru_end)))
        count = tru_end - start


        #Create a list of values for the number of each sample
        nums = [pbs_num, pbs_jj_num, bank_num, bank_jj_num, rna_num, rna_jj_num, fmt_num,
             fmt_jj_num, stool_jj_num, pH_num]

        #list of sample types
        sample_types = ['PBS', 'PBS', 'Bank', 'Bank', 'RNA', 'RNA', 'FMT', 'FMT', 'Stool', 'pH']

        #Create list of samples_types with aliquot number
        Sample_Ty = []

        for x in range(0, count):
            for sample in zip(sample_types, nums):
                if sample[1] > 10:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i:02}')
                else:
                    for i in range(1, sample[1]+1):
                        Sample_Ty.append(f'{sample[0]}-{i}')

        # list of owners in the correct orientation for samples
        owner_types = ['PB', 'JJ-92108', 'PB', 'JJ-92108', 'PB', 'JJ-92108', 'PB', 'JJ-92108', 'JJ-92108', 'PB']

        #create list of owners
        owner_tot = []
        for x in range(0, count):
            for sample in zip(owner_types, nums):
                for i in range(sample[1]):
                    owner_tot.append(f'{sample[0]}')


        # adding 'PBT' to lab_pbt label

        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot_list))

        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)



        # 2d barcode
        qr=[]
        for i in range(0, len(lab_pbt)):
            str = ""
            str = lab_pbt[i] + " " + Sample_Ty[i]
            qr.append(str)
        print('QR Code: \n',qr)


        #create table
        Header = ['QR_code', 'Line1', 'Line2', 'Line3']

        with open('Stool Label.csv', 'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(Header)

            for i in range(0,len(lab_pbt)):
                #print(qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i])
                rowDaTa= [qr[i], lab_pbt[i],Sample_Ty[i],owner_tot[i]]
                csv_writer.writerow(rowDaTa)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.read_csv("Stool Label.csv").sort_index(ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        df2 = df2.rename(columns={ 'QR_code' : 'QR_code-2', 'Line1' : 'Line1_2', 'Line2' : 'Line2_2' , 'Line3' : 'Line3_2'})

        label_sheet = df2.join(df1, how='right')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} COH stool label 2up.csv'


        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download COH Fecal CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif analysis_type == 'poo kit labels':
        # Code for Analysis 3
        st.write("Running CRCi Fecal...")
        st.write(f"Start Value: {start_value}")
        st.write(f"End Value: {end_value}")

        # Generate DataFrame
        #Enter Start and End PBT numbers (numbers only)
        start = start_value
        end = end_value
        copies =  14

        # Generate list of just the # in PBT_# and count the total number of sets of labels
        tru_end = (end) + 1
        num_tot = list(itertools.chain.from_iterable(itertools.repeat(x, copies) for x in range(start, tru_end)))
        count = tru_end - start


        #add leading 0's for PBT number
        formatter = "{:05d}".format
        num_tot_list = list(map(formatter, num_tot))

        # adding label to ID
        lab_pbt = []
        for value in num_tot_list:
            pbt = f'PBT-{value}'
            lab_pbt.append(pbt)


        # seperate labels for 2-up. Odd labels print on top, even on bottom.
        df = pd.DataFrame(lab_pbt, columns = ['PBT_Top']).sort_values(by='PBT_Top', ascending=False)
        df1 = df.iloc[0::2].reset_index().drop(columns=['index'])
        df2 = df.iloc[1::2].reset_index().drop(columns=['index'])

        #Rename df2 columns for the subsequent join command
        df2 = df2.rename(columns={ 'PBT_Top' : 'PBT_Bottom'})

        #Join df2 to df1. If there's an odd number of samples the "how" flag is important
        label_sheet = df1.join(df2, how='left')

        filename = f'{lab_pbt[0] + " to " + lab_pbt[-1]} kit labels-2up.csv'


        # Download CSV
        csv_file = label_sheet.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download poo kit labels CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
