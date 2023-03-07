# -------------------------------
# SpheroidSizer.py
# Author: Felix Romer
# Email: felix.lucas.romer@gmail.com
# Web: https://github.com/DaRoemer/SpheroidSizer
# Date: 2023/03/07
# Version: 1
# Last Change: 2023/03/07
# Description: Find Spheroids automatically or manuel and calculate area, roundness as well as minor and major axis length.
#              If a folder structure with one folder for the experiment with subfolder for each day / condition / or other is supplied, 
#              the output is saved as a excel file with each subfolder as a separate sheet and each image as a row in the sheet
# -------------------------------

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import cv2


points=[]
resized_img = np.array([])
use_img = np.array([])


def automatic_spheroid_finder(image_path, Outfile):
    # read the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    resized_gray_img = scale_img(gray, 50)

    # threshold using Otsu's method
    ret, thresh = cv2.threshold(resized_gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #invert tresh
    inv_thresh = cv2.bitwise_not(thresh)

    # Find connected components and their statistics
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_thresh, connectivity=8)

    # Define minimum area threshold
    min_area = 100 # Adjust this value as needed

    # Calculate center of image
    height, width = inv_thresh.shape
    center = np.array([width/2, height/2])

    # Find index of component closest to center of image for components > min_area
    min_dist = np.inf
    min_dist_idx = -1
    for i in range(1, n_labels):
        # cv2.imshow(f'{i}', labels[i])
        centroid = centroids[i]
        dist = np.linalg.norm(center - centroid)
        if (dist < min_dist) and (stats[i, cv2.CC_STAT_AREA] > min_area):
            min_dist = dist
            min_dist_idx = i

    # Create binary image containing only the closest component
    closest_img = np.uint8(labels == min_dist_idx) * 255


    # Find contours of all objects in the binary image
    contours, hierarchy = cv2.findContours(closest_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # Calculate area
    area = cv2.contourArea(cnt)

    # Calculate roundness
    perimeter = cv2.arcLength(cnt,True)
    roundness = 4*np.pi*area/perimeter**2

    # Find the centroid and moments of the object
    M = cv2.moments(closest_img)

    # Calculate the covariance matrix
    coords = np.column_stack(np.where(closest_img > 0))
    covariance_matrix = np.cov(coords, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Get the major and minor axis lengths
    major_axis_length = 4 * np.sqrt(eigenvalues[1])
    minor_axis_length = 4 * np.sqrt(eigenvalues[0])

    # Draw contours on resized image
    result = cv2.cvtColor(resized_gray_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 1)

    # Display result
    continue_mauel=False

    while True:
        cv2.imshow('Result - Press ENTER to close, Press M to initiate manual analysis', result)
        key = cv2.waitKey(1)
        if key == 13:
            cv2.imwrite(Outfile, result) #enter
            break
        elif key == 109 or key == 77: #m M
            continue_mauel=True
            break
    
    cv2.destroyAllWindows()

    return area, roundness, major_axis_length, minor_axis_length, 'automatic', continue_mauel


### ---------



def draw_lines(points, image):
    # draw lines between each pair of adjacent points
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)
    # connect the last point to the first point

def mouse_callback(event, x, y, flags, params):
    global resized_img, use_img, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) > 1:
            draw_lines(points, use_img)
        cv2.circle(use_img, (x, y), 2, (0, 0, 255), 2)
        cv2.imshow('Manual Analysis - Left Mouse to set point / right Mouse to remove / ENTER to accept points / Q to quit', use_img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        use_img = resized_img.copy()
        points.pop()
        
        draw_lines(points, use_img)
        for point in points:
            cv2.circle(use_img, point, 2, (0, 0, 255), 2)
        cv2.imshow('Manual Analysis - Left Mouse to set point / right Mouse to remove / ENTER to accept points / Q to quit', use_img)

def scale_img(image, scale_percent=50):

    height, width = image.shape

    
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def minor_major_axis(coord):
    # Compute the centroid of the polygon
    centroid = np.mean(coord, axis=0)

    # Translate the polygon to the origin
    coord -= centroid

    # Compute the covariance matrix
    cov = np.cov(coord, rowvar=False)

    # Find the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov)

    # Sort the eigenvalues in decreasing order
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Extract the major and minor axis lengths
    major_axis = 2.0 * np.sqrt(evals[0])
    minor_axis = 2.0 * np.sqrt(evals[1])

    print(f"Major axis: {major_axis}")
    print(f"Minor axis: {minor_axis}")
    return major_axis, minor_axis

def manuel_drawing(file_name, Outfile):
    global resized_img, use_img, points
    points=[]
    image = cv2.imread(file_name)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        pass
    resized_img = scale_img(image)
    use_img = resized_img.copy()
    print('Please select the edge manually with a LEFT click.')
    print('RIGHT click removes the last point')
    print('Press ENTER is finished. The last and first point will be connected automatically')
    print('Press Q to quit')
    cv2.namedWindow('Manual Analysis - Left Mouse to set point / right Mouse to remove / ENTER to accept points / Q to quit', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Manual Analysis - Left Mouse to set point / right Mouse to remove / ENTER to accept points / Q to quit', mouse_callback)

    while True:
        cv2.imshow('Manual Analysis - Left Mouse to set point / right Mouse to remove / ENTER to accept points / Q to quit', use_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 13:
            print(key, points)  # Enter key
            if len(points) > 2:  # need at least 3 points to form a polygon
                cv2.line(use_img, points[-1], points[0], (0, 255, 0), 2)
                cv2.imwrite(Outfile, use_img)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                break

    area = cv2.contourArea(np.array(points))
    print('Area manuel:', area * 2)
    perimeter = cv2.arcLength(np.array(points), True)
    roundness = (4 * np.pi * area) / perimeter**2
    print('Roundness manuel: ', roundness)
    major_axis, minar_axis = minor_major_axis(points)

    return area, roundness, major_axis, minar_axis, 'manuel'

def analyse_one_image(filename, dir_to_folder, df):
    image_path = os.path.join(dir_to_folder, filename)

    outhead, _ = image_path.rsplit('.', 1)
    Outfile = outhead+'_detected_Spheroid.tif'
    print('Save in ', Outfile)

    swithc_to_manuel=False
    valid_spheroid=True
    try:
        area, roundness, major_axis_len, minor_axis_length, mode, swithc_to_manuel = automatic_spheroid_finder(image_path, Outfile)


        print('Area: ', area)
        print('Roundness: ', roundness)
        print('ENTER to continue')
        print('M to draw the outlines manually')

    except:
        print('Automatic analysis failed. Increase contrast or continue with automatic processing')
        swithc_to_manuel=True

    if swithc_to_manuel==True:

        print('manuel')
        area, roundness, major_axis_len, minor_axis_length, mode = manuel_drawing(image_path, Outfile)

    if valid_spheroid == False:
        area, roundness, major_axis_len, minor_axis_length, mode = manuel_drawing(image_path, Outfile)
    
    new_row = {'Folder':dir_to_folder,'File': filename,	'Area': area, 'Length': major_axis_len,	'Width': minor_axis_length, 'Roundness':roundness, 'Roundness_width/length':minor_axis_length/major_axis_len, 'mode':mode}
    df.loc[len(df)] = new_row
    return df

# x='C:/Users/felix_9ny56v1/.01Felix/01_Studium/01_Bachelor_HSMannheim/HIWI/Test/Marker_500_Mikrometer.tif'
# main(x)



print('Please select the partenfolder. Ever subfolder should represent day/replicate/etc All Images in this folder will be processed.')
print('A window may open in the background')
root = tk.Tk()
root.withdraw()

mother_folder = filedialog.askdirectory()    # window may open in the background
folders = [f for f in os.listdir(mother_folder) if os.path.isdir(os.path.join(mother_folder, f))]

print('The folder is: ', mother_folder)

# loop through all images in the folder
# Create a Pandas Excel writer using XlsxWriter as the engine.
exel_file=mother_folder+'/output.xlsx'
writer = pd.ExcelWriter(exel_file, engine='xlsxwriter')
for folder in folders:
    folder=mother_folder+'/'+folder
    df =pd.DataFrame(columns=['Folder',	'File',	'Area', 'Length',	'Width', 'Roundness', 'Roundness_width/length', 'mode', 'commend'])
    for filename in os.listdir(folder):
        
        resized_img = np.array([])
        use_img = np.array([])
        df=analyse_one_image(filename, folder, df)

    # Write each dataframe to a different worksheet.
    df.to_excel(writer, sheet_name=folder)


# Close the Pandas Excel writer and output the Excel file.
writer.save()


                
    



