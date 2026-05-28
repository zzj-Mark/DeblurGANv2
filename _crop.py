import cv2  
img=cv2.imread('test_img/kohler_visual.png')  
h,w=img.shape[:2]  
ph,pw=h//4,w//2  
[cv2.imwrite(f'test_img/kohler_blur_{i+1}.png',img[i*ph:(i+1)*ph,:pw]) for i in range(4)]  
print('done')  
