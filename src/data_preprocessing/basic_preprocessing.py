import os

import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Set, Any

from settings.settings import BASE_DIR


def adjust_target_variable_labels(df: DataFrame):
    lm = {
        "Cognitively Normal": ["Cognitively normal", "No dementia"],
        "Uncertain Dementia": ["uncertain dementia", "Unc: ques. Impairment", "uncertain- possible NON AD dem",
                               "0.5 in memory only", "uncertain  possible NON AD dem", "Incipient demt PTP",
                               "Unc: impair reversible", "Incipient Non-AD dem"],
        "Alzheimer Dementia": ["AD dem w/depresss- not contribut", "AD Dementia", "AD dem distrubed social- with",
                               "AD dem w/CVD contribut", "AD dem visuospatial- prior", "AD dem Language dysf after",
                               "AD dem w/PDI after AD dem not contrib", "AD dem distrubed social- prior",
                               "AD dem distrubed social- after", "AD dem w/PDI after AD dem contribut",
                               "AD dem w/depresss  not contribut", "AD dem w/oth (list B) contribut",
                               "AD dem w/depresss- contribut", "AD dem w/oth (list B) not contrib",
                               "AD dem w/CVD not contrib",
                               "AD dem Language dysf prior", "AD dem Language dysf with",
                               "AD dem w/oth unusual features",
                               "AD dem cannot be primary", "AD dem w/oth unusual feat/subs demt",
                               "AD dem w/depresss  contribut", "AD dem visuospatial, after",
                               "AD dem visuospatial- with", "AD dem w/oth unusual features/demt on",
                               "AD dem w/Frontal lobe/demt at onset", "AD dem/FLD prior to AD dem",
                               "AD dem w/depresss, not contribut"],
        "Non AD Dementia":    ["Non AD dem- Other primary", "Vascular Demt- primary", "Frontotemporal demt. prim",
                               "DLBD- primary", "Dementia/PD- primary", "DAT", "Vascular Demt  primary", "DLBD- secondary",
                               "ProAph w/o dement", "Vascular Demt- secondary", "DLBD, primary",
                               "DAT w/depresss not contribut"]
    }

    label_map = {}
    for key, value in lm.items():
        for v in value:
            label_map[v] = key

    # Replace values in DataFrame
    df.replace({"dx1": label_map}, inplace=True)


def remove_others(df: DataFrame, columns: Set[Any]):
    cols_total: Set[Any] = set(df.columns)
    diff: Set[Any] = cols_total - columns
    return df.drop(diff, axis=1)


def load_data():
    adrc = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "ADRC Clinical Data.csv"))

    '''
        Replacing '.' (missing) values with numpy nan
    '''
    adrc.replace(to_replace='.', value=np.nan, inplace=True)

    '''
        Dropping irrelevant labels from tables 
    '''
    adrc.drop(
        labels=["Date", "Age", "acsparnt", "height", "weight", "primStudy", "acsStudy"],
        axis="columns", inplace=True)


    adrc["ADRC_ADRCCLINICALDATA ID"] = adrc["ADRC_ADRCCLINICALDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    adjust_target_variable_labels(adrc)

    clinician_diagnosis = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "Clinician Diagnosis.csv"))
    clinician_diagnosis.drop(
        labels=["Date", "Age", "DEMENTED", "MCIAMEM", "MCIAPLUS", "MCIAPLAN", "MCIAPATT", "MCIAPEX",
                "MCIAPVIS", "MCINON1", "MCIN1LAN", "MCIN1ATT", "MCIN1EX", "MCIN1VIS", "MCINON2", "MCIN2LAN",
                "MCIN2ATT", "MCIN2EX", "MCIN2VIS", "IMPNOMCI", "PROBAD", "PROBADIF", "POSSAD", "POSSADIF",
                "DLB", "DLBIF", "VASC", "VASCIF", "VASCPS", "VASCPSIF", "ALCDEM", "ALCDEMIF", "DEMUN",
                "DEMUNIF", "FTD", "FTDIF", "PPAPH", "PPAPHIF", "PNAPH", "SEMDEMAN", "SEMDEMAG", "PPAOTHR",
                "PSPIF", "CORTIF", "HUNTIF", "PRIONIF", "MEDSIF", "DYSILLIF", "DEPIF", "OTHPSYIF", "DOWNSIF",
                "PARKIF", "STROKIF", "HYCEPHIF", "BRNINJIF", "NEOPIF", "COGOTHX", "COGOTHIF", "COGOTH2X",
                "COGOTH2F", "COGOTH3X", "COGOTH3F"],
        axis="columns", inplace=True)

    clinician_diagnosis["UDS_D1DXDATA ID"] = clinician_diagnosis["UDS_D1DXDATA ID"]\
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    # clinician_judgements = pd.read_csv("../numerical_data/raw/Clinician Judgements.csv")
    #
    # clinician_judgements.drop(labels=["Date", "Age", "DECAGE", "COGMEM", "COGJUDG", "COGLANG", "COGVIS", "COGATTN"],
    #                           axis="columns", inplace=True)
    # only 3 columns have enough values, all others are over 50% missing

    faqs = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "FAQs.csv"))

    faqs.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    faqs["UDS_B7FAQDATA ID"] = faqs["UDS_B7FAQDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    freesurfers = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "FreeSurfers.csv"))

    freesurfers.drop(
        labels=["FS Date", "Included T1s"],
        axis="columns", inplace=True)

    # freesurfers["UDS_D1DXDATA ID"] = freesurfers["UDS_D1DXDATA ID"] \
    #     .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    gds =pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "GDS.csv"))

    gds.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    gds["UDS_B6BEVGDSDATA ID"] = gds["UDS_B6BEVGDSDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    his_and_cvd = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "HIS and CVD.csv"))

    his_and_cvd.drop(
        labels=["Date", "Age", "CVDIMAG1", "CVDIMAG2", "CVDIMAG3", "CVDIMAG4", "CVDIMAGX"],
        axis="columns", inplace=True)

    his_and_cvd["UDS_B2HACHDATA ID"] = his_and_cvd["UDS_B2HACHDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    npi_q = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "NPI-Q.csv"))

    npi_q.drop(
        labels=["Date", "Age", "INITIALS", "NPIQINFX", "DELSEV", "HALLSEV", "AGITSEV", "DEPDSEV", "ANXSEV", "ELATSEV",
                "APASEV", "DISNSEV", "IRRSEV", "MOTSEV", "NITESEV", "APPSEV"],
        axis="columns", inplace=True)

    npi_q["UDS_B5BEHAVASDATA ID"] = npi_q["UDS_B5BEHAVASDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    # TODO: maybe sev is severity and is present only for patients who have the problem (fill with zeros or smth)?

    physical_neuro_findings = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "Physical Neuro Findings.csv"))

    physical_neuro_findings.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    physical_neuro_findings["UDS_B8EVALDATA ID"] = physical_neuro_findings["UDS_B8EVALDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    sub_health_history = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "Subject Health History.csv"))

    sub_health_history.drop(
        labels=["Date", "Age", "EXAMDATE", "CVOTHRX", "STROK1YR", "STROK2YR", "STROK3YR", "STROK4YR", "STROK5YR",
                "STROK6YR", "TIA1YR", "TIA2YR", "TIA3YR", "TIA4YR", "TIA5YR", "TIA6YR", "CBOTHRX", "PDYR", "PDOTHRYR",
                "NCOTHRX", "ABUSX", "PSYCDISX"],
        axis="columns", inplace=True)

    sub_health_history["UDS_A5SUBHSTDATA ID"] = sub_health_history["UDS_A5SUBHSTDATA ID"] \
        .map(lambda x: x.split("_")[0] + "_" + x.split("_")[2])

    subjects = pd.read_csv(os.path.join(BASE_DIR, "data", "numerical_data", "raw", "subjects.csv"))

    subjects.drop(
        labels=["YOB"],
        axis="columns", inplace=True)

    '''
        Creating a list of ids for each table
    '''
    adrc_ids = adrc["ADRC_ADRCCLINICALDATA ID"].tolist()
    clinician_diagnosis_ids = clinician_diagnosis["UDS_D1DXDATA ID"].tolist()
    faqs_ids = faqs["UDS_B7FAQDATA ID"].tolist()
    gds_ids = gds["UDS_B6BEVGDSDATA ID"].tolist()
    his_and_cvd_ids = his_and_cvd["UDS_B2HACHDATA ID"].tolist()
    npi_q_ids = npi_q["UDS_B5BEHAVASDATA ID"].tolist()
    physical_neuro_findings_ids = physical_neuro_findings["UDS_B8EVALDATA ID"].tolist()
    sub_health_history_ids = sub_health_history["UDS_A5SUBHSTDATA ID"].tolist()

    all_ids = [clinician_diagnosis_ids, faqs_ids, gds_ids, his_and_cvd_ids,
               npi_q_ids, physical_neuro_findings_ids, sub_health_history_ids]

    '''
        Removing all instances from 4089 tables that don't contain labels in adrc table 
    '''

    # map where key is a column name which contains ids, and value is dataframe
    id_dataframe_map = {
        "UDS_D1DXDATA ID": clinician_diagnosis,
        "UDS_B7FAQDATA ID": faqs,
        "UDS_B6BEVGDSDATA ID": gds,
        "UDS_B2HACHDATA ID": his_and_cvd,
        "UDS_B5BEHAVASDATA ID": npi_q,
        "UDS_B8EVALDATA ID": physical_neuro_findings,
        "UDS_A5SUBHSTDATA ID": sub_health_history
    }

    seti = set()
    for i in range(len(all_ids[1])):
        if all_ids[1][i] not in adrc_ids:
            seti.add(all_ids[1][i])

    for k, v in id_dataframe_map.items():
        criterion = v[k].map(lambda i: i not in seti)
        id_dataframe_map[k] = v[criterion]

    adrc = adrc[adrc["dx1"].notnull()]

    # number of rows in each csv
    # adrc 6224 others 4089 freesurfers 1984 subjects 1098
    # all tables with 4089 have all the same instances (ids are all the same)

    data = {"adrc": adrc,
            "clinician_diagnosis": id_dataframe_map["UDS_D1DXDATA ID"],
            "faqs": id_dataframe_map["UDS_B7FAQDATA ID"],
            "freesurfers": freesurfers,
            "gds": id_dataframe_map["UDS_B6BEVGDSDATA ID"],
            "his_and_cvd": id_dataframe_map["UDS_B2HACHDATA ID"],
            "npi_q": id_dataframe_map["UDS_B5BEHAVASDATA ID"],
            "physical_neuro_findings": id_dataframe_map["UDS_B8EVALDATA ID"],
            "sub_health_history": id_dataframe_map["UDS_A5SUBHSTDATA ID"],
            "subjects": subjects}

    data["adrc"].rename(columns={"ADRC_ADRCCLINICALDATA ID": "ID"}, inplace=True)
    data["clinician_diagnosis"].rename(columns={"UDS_D1DXDATA ID": "ID"}, inplace=True)
    data["faqs"].rename(columns={"UDS_B7FAQDATA ID": "ID"}, inplace=True)
    data["gds"].rename(columns={"UDS_B6BEVGDSDATA ID": "ID"}, inplace=True)
    data["his_and_cvd"].rename(columns={"UDS_B2HACHDATA ID": "ID"}, inplace=True)
    data["npi_q"].rename(columns={"UDS_B5BEHAVASDATA ID": "ID"}, inplace=True)
    data["physical_neuro_findings"].rename(columns={"UDS_B8EVALDATA ID": "ID"}, inplace=True)
    data["sub_health_history"].rename(columns={"UDS_A5SUBHSTDATA ID": "ID"}, inplace=True)
    data["freesurfers"].rename(columns={"FS_FSDATA ID": "ID"}, inplace=True)

    '''
        Removing subject column from every dataframe but one, so it doesnt repeat in merged dataframe
    '''

    for key in ["faqs", "gds", "his_and_cvd", "npi_q", "physical_neuro_findings", "sub_health_history"]:
        data[key].drop(
                       labels=["Subject"],
                       axis="columns", inplace=True)

    id_dx1 = remove_others(data["adrc"], {"ID", "dx1"})

    merged = pd.merge(data["clinician_diagnosis"], data["faqs"], on="ID", how="inner")
    merged = pd.merge(merged, data["gds"], on="ID", how="inner")
    merged = pd.merge(merged, data["his_and_cvd"], on="ID", how="inner")
    merged = pd.merge(merged, data["npi_q"], on="ID", how="inner")
    merged = pd.merge(merged, data["physical_neuro_findings"], on="ID", how="inner")
    merged = pd.merge(merged, data["sub_health_history"], on="ID", how="inner")
    merged = pd.merge(merged, id_dx1, on="ID", how="left")
    data["merged"] = merged

    return data


if __name__ == '__main__':
    data = load_data()
