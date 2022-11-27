from attr import field
import streamlit as st
import numpy as np
import cv2
import time
import Chuong3 as chuong3
import Chuong4 as chuong4
import Chuong9 as chuong9
import Buoc1_Get_face_NhanDangKhuonMat_Facebook as buoc1_faebook
import Buoc2_Training_NhanDangKhuonMat_Facebook as buoc2_faebook
import Buoc3_Predict_NhanDangKhuonMat_Facebook as buoc3_faebook
import object_detection as object_detection

#Thiết lập thông tin tiêu đề của sidebar
st.sidebar.title("Thu Hương - Đồ án học phần")

#Thiết lập menu_page 
menu_page = st.sidebar.selectbox('Mời chọn menu xử lý', ['❄️ Mở hình ảnh starry_night', '❄️ Mở máy ảnh và lưu file' , '❄️ Nhận diện Face mở bằng Camera', '❄️ Nhận diện face PP face detect opencv dnn', '❄️ Nhận diện face PP face detect opencv dnn_caffe', '❄️ Nhận diện object detect opencv yolo3', '❄️ Nhận diện NhanDangKhuonMat Facebook', '❄️ Chọn thuật toán xử lý ảnh'])   

#Thiết lập backgroud bên trái cho sidebar
st.sidebar.markdown(
   f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url(https://images.unsplash.com/photo-1555178364-6c1f870e3349?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1288&q=80);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Thiết lập background bên phải cho sidebar
st.markdown(
    f'''<style>
    .stApp {{
    background-image: url(https://images.unsplash.com/photo-1483137140003-ae073b395549?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80);
    background-size: cover;
    }}
    </style>''', 
    unsafe_allow_html=True
)
        
#Thiết lập hàm xử lý ảnh chương 3
def XuLyAnh_Chuong3(name_type, image_in):
    chuong3_Title = ''
    image_title = ''
    image_out = np.empty

    #Thiết lập image_out
    try: 
        image_out = np.zeros(image_in.shape, np.uint8)
    except: 
        except_Title = '<p style="font-family: Georgia; color: #4A235A; font-size: 40px; font-weight: bold; text-align: center;">Hình ảnh không thực hiện chuyển đổi được</p>'
        st.markdown(except_Title, unsafe_allow_html=True) 
        return

    #Kiểm tra name_type tính năng nào được xử lý
    if name_type == 'Negative':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 1 _ Negative</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung 
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Negative: Hàm làm âm ảnh. Kết quả hàm âm ảnh: Trắng thành đen, đen thành trắng</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Negative(image_in, image_out)
    elif name_type == 'Logarit':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 2 _ Logarit</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)  
        #Thiết lập nội dung  
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Logarit: Hàm xử lý ảnh bằng phương pháp Logarit. Kết quả hàm Logarit: Sáng ít thì làm cho sáng nhiều, Đen nhiều thì thành đen ít</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)    
        #Gọi hàm xử lý
        image_out = chuong3.Logarit(image_in, image_out)
    elif name_type == 'Power':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 3 _ Power</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung  
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Power: Hàm xử lý ảnh bằng phương pháp lũy thừa ảnh. Kết quả hàm Power: Làm cho ảnh trở trên tối hơn, hay sáng hơn đều được</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Power(image_in, image_out)
    elif name_type == 'PiecewiseLinear':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 4 _ PiecewiseLinear</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu PiecewiseLinear: Hàm xử lý ảnh bằng phương pháp Piecewise. Kết quả hàm PiecewiseLinear: Kéo dài độ tương phản, phạm vi mức cường độ làm sáng hơn</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.PiecewiseLinear(image_in, image_out)
    elif name_type == 'Histogram':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 5 _ Histogram</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu HistogramEqualization: Hàm xử lý ảnh bằng phương pháp Histogram. Kết quả biểu Histogram: là dạng biểu đồ thể hiện tần suất theo dạng cột theo dữ liệu tương ứng, làm cho ảnh đẹp hơn</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Histogram(image_in, image_out)
    elif name_type == 'HistogramEqualization':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 6 _ HistogramEqualization</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu HistogramEqualization: Hàm xử lý ảnh bằng phương pháp HistogramEqualization. Kết quả biểu HistogramEqualization: là dạng biểu đồ thể hiện tần suất theo dạng cột theo dữ liệu tương ứng, làm cho ảnh đẹp hơn, hàm HistogramEqualization tốt hơn Histogram</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.HistogramEqualization(image_in, image_out)
    elif name_type == 'LocalHistogram':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 7 _ LocalHistogram</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu LocalHistogram: Hàm xử lý ảnh bằng phương pháp LocalHistogram. Kết quả hàm LocalHistogram: Làm rõ 1 vùng trong ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.LocalHistogram(image_in, image_out)
    elif name_type == 'HistogramStatistics':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 8 _ HistogramStatistics</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu HistogramStatistics: Hàm xử lý ảnh bằng phương pháp HistogramStatistics. Kết quả hàm HistogramStatistics: Thống kê 1 vùng trong ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.HistogramStatistics(image_in, image_out)
    elif name_type == 'Smoothing':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 9 _ Smoothing</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Smoothing: Hàm xử lý ảnh bằng phương pháp Smoothing. Kết quả hàm Smoothing: Làm trơn, làm nhòe ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Smoothing(image_in)
    elif name_type == 'SmoothingGauss':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 10 _ SmoothingGauss</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu SmoothingGauss: Hàm xử lý ảnh bằng phương pháp SmoothingGauss. Kết quả hàm SmoothingGauss: Làm trơn, làm nhòe ảnh. Thuật toán SmoothingGauss tốt hơn thuật toán Smooth cơ bản</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.SmoothingGauss(image_in)
    elif name_type == 'MedianFilter':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 11 _ MedianFilter</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu MedianFilter: Hàm xử lý ảnh bằng phương pháp MedianFilter. Kết quả hàm MedianFilter: Lọc nhiễu ảnh, lọc các điểm nhiễu ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.MedianFilter(image_in, image_out)
    elif name_type == 'Sharpen':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 12 _ Sharpen</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Sharpen: Hàm xử lý ảnh bằng phương pháp Sharpen. Kết quả hàm Sharpen: Làm nét, bén nhọn các góc của ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Sharpen(image_in)
    elif name_type == 'UnSharpMasking':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 13 _ UnSharpMasking</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu UnSharpMasking: Hàm xử lý ảnh bằng phương pháp UnSharpMasking. Kết quả hàm UnSharpMasking: Làm nét ảnh bằng phương pháp UnSharpMasking</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.UnSharpMasking(image_in)
    elif name_type == 'Gradient':
        #Thiết lập tiêu đề 
        chuong3_Title = '<p style="font-family: Georgia; color: #351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 14 _ Gradient</p>'
        st.markdown(chuong3_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong3_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Gradient: Hàm xử lý ảnh bằng phương pháp Gradient. Kết quả hàm Gradient: Đạo hàm ống kính tách biên của ảnh</p>'
        st.markdown(chuong3_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong3.Gradient(image_in)
    
    #Hiển thị thông tin image_in
    image_title = '<p style="font-family: Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;"> Hình '+ name_type +' trước khi thay đổi : </p>'
    st.markdown(image_title, unsafe_allow_html=True)   
    st.image(image_in, width=300)
    
    #Hiển thị thông tin image_out
    image_title = '<p style="font-family:Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;">Hình '+ name_type +' sau khi thay đổi :</p>'
    st.markdown(image_title, unsafe_allow_html=True)      
    st.image(image_out,  width=300)

#Thiết lập hàm xử lý ảnh chương 4
def XuLyAnh_Chuong4(name_type, image_in):
    chuong4_Title = ''
    image_title = ''
    image_out = np.empty

    #Thiết lập image_out
    try: 
        image_out = np.zeros(image_in.shape, np.uint8)
    except: 
        except_Title = '<p style="font-family: Georgia; color: #4A235A; font-size: 40px; font-weight: bold; text-align: center;">Hình ảnh không thực hiện chuyển đổi được</p>'
        st.markdown(except_Title, unsafe_allow_html=True) 
        return

    #Kiểm tra name_type tính năng nào được xử lý
    if name_type == 'Spectrum':
        #Thiết lập tiêu đề 
        chuong4_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 1 _ Spectrum</p>'
        st.markdown(chuong4_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong4_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;;">Giới thiệu Spectrum: Hàm xử lý ảnh bằng phương pháp Spectrum. Kết quả hàm Spectrum: Xử lý quang phổ bằng các công thức tần số sóng điện từ, ánh sánh, điểm ảnh</p>'
        st.markdown(chuong4_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong4.Spectrum(image_in)
    elif name_type == 'FrequencyFilter':
        #Thiết lập tiêu đề 
        chuong4_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 2 _ FrequencyFilter</p>'
        st.markdown(chuong4_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong4_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu FrequencyFilter: Hàm xử lý ảnh bằng phương pháp FrequencyFilter. Kết quả hàm FrequencyFilter: Bộ lọc cho tín hiệu có tần số thấp hơn tần số cắt đã chọn và làm suy giảm tín hiệu có tần số cao hơn tần số cắt để xử lý ảnh</p>'
        st.markdown(chuong4_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong4.FrequencyFilter(image_in)
    elif name_type == 'DrawFilter':
        #Thiết lập tiêu đề 
        chuong4_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 3 _ DrawFilter</p>'
        st.markdown(chuong4_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong4_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu DrawFilter: Hàm xử lý ảnh bằng phương pháp DrawFilter. Kết quả hàm DrawFilter: Xử lý lọc ảnh cơ bản</p>'
        st.markdown(chuong4_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong4.DrawFilter(image_in)
    elif name_type == 'RemoveMoire':
        #Thiết lập tiêu đề 
        chuong4_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 4 _ RemoveMoire</p>'
        st.markdown(chuong4_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong4_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu RemoveMoire: Hàm xử lý ảnh bằng phương pháp Moire. Kết quả hàm RemoveMoire: Xóa các điểm nhiễu ảnh</p>'
        st.markdown(chuong4_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong4.RemoveMoire(image_in)
   
    #Hiển thị thông tin image_in
    image_title = '<p style="font-family: Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;">Hình '+ name_type +' trước khi thay đổi :</p>'
    st.markdown(image_title, unsafe_allow_html=True)   
    st.image(image_in, width=300)
    
    #Hiển thị thông tin image_out
    image_title = '<p style="font-family:Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;">Hình '+ name_type +' sau khi thay đổi :</p>'
    st.markdown(image_title, unsafe_allow_html=True)      
    st.image(image_out,  width=300)

#Thiết lập hàm xử lý ảnh chương 9
def XuLyAnh_Chuong9(name_type, image_in):
    chuong9_Title = ''
    image_title = ''
    image_out = np.empty

    #Thiết lập image_out
    try: 
        image_out = np.zeros(image_in.shape, np.uint8)
    except: 
        except_Title = '<p style="font-family: Georgia; color: #4A235A; font-size: 40px; font-weight: bold; text-align: center;">Hình ảnh không thực hiện chuyển đổi được</p>'
        st.markdown(except_Title, unsafe_allow_html=True) 
        return

    #Kiểm tra name_type tính năng nào được xử lý
    if name_type == 'Erosion':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 1 _ Erosion</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Erosion: Hàm xử lý ảnh bằng phương pháp Erosion. Kết quả hàm Erosion: Làm bào mòn ảnh</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.Erosion(image_in, image_out)
    elif name_type == 'Dilation':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 2 _ Dilation</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Dilation: Hàm xử lý ảnh bằng phương pháp Dilation. Kết quả hàm Dilation: Làm giãn nở chữ, làm mập ảnh</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.Dilation(image_in, image_out)
    elif name_type == 'OpeningClosing':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 3 _OpeningClosing</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu OpeningClosing: Hàm xử lý ảnh bằng phương pháp OpeningClosing. Kết quả hàm OpeningClosing: Dùng để xóa nhiễu ảnh, có thể thay thế bằng các phương pháp như: Spectrum, FrequencyFilter, DrawFilter, RemoveMoire của chương 4</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.OpeningClosing(image_in, image_out)
    elif name_type == 'Boundary':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 4 _ Boundary</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu Boundary: Hàm xử lý ảnh bằng phương pháp Boundary. Kết quả hàm Boundary: Phát hiện diểm biên, khoanh vùng điểm biên của ảnh</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.Boundary(image_in)
    elif name_type == 'HoleFill':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 5 _ HoleFill</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu HoleFill: Hàm xử lý ảnh bằng phương pháp HoleFill. Kết quả hàm HoleFill: Lấp 1 lỗ trống trên ảnh/p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.HoleFill(image_in)
    elif name_type == 'HoleFillMouse':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 6 _ HoleFillMouse</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu HoleFillMouse: Hàm xử lý ảnh bằng phương pháp HoleFillMouse. Kết quả hàm HoleFillMouse: Lấp lỗ trống bằng chuột</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.HoleFill(image_in)
    elif name_type == 'MyConnectedComponent':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 7 _ MyConnectedComponent</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu MyConnectedComponent: Hàm xử lý ảnh bằng phương pháp MyConnectedComponent. Kết quả hàm MyConnectedComponent: thành phần liên thông, đếm có bao nhiêu miếng xương gà</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.MyConnectedComponent(image_in)
    elif name_type == 'ConnectedComponent':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 8 _ ConnectedComponent</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu ConnectedComponent: Hàm xử lý ảnh bằng phương pháp ConnectedComponent. Kết quả hàm ConnectedComponent: Thành phần liên thông, đếm có bao nhiêu miếng xương gà, tách ảnh</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.ConnectedComponent(image_in)
    elif name_type == 'CountRice':
        #Thiết lập tiêu đề 
        chuong9_Title = '<p style="font-family: Georgia; color:#351F59; font-size: 40px; font-weight: bold; text-align: center;">Câu 9 _ CountRice</p>'
        st.markdown(chuong9_Title, unsafe_allow_html=True)   
        #Thiết lập nội dung
        chuong9_Content = '<p style="font-family: Times; color: #000000; font-size: 20px; text-align: justify;">Giới thiệu CountRice: Hàm xử lý ảnh bằng phương pháp CountRice. Kết quả hàm CountRice: Đếm có bao nhiêu hạt gạo</p>'
        st.markdown(chuong9_Content, unsafe_allow_html=True)   
        #Gọi hàm xử lý
        image_out = chuong9.CountRice(image_in)
   
    #Hiển thị thông tin image_in
    image_title = '<p style="font-family: Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;">Hình '+ name_type +' trước khi thay đổi :</p>'
    st.markdown(image_title, unsafe_allow_html=True)   
    st.image(image_in, width=300)
    
    #Hiển thị thông tin image_out
    image_title = '<p style="font-family:Georgia; color: #1B4F72; font-size: 20px; font-weight: bold;">Hình '+ name_type +' sau khi thay đổi :</p>'
    st.markdown(image_title, unsafe_allow_html=True)      
    st.image(image_out,  width=300)

#Thiết lập hàm OpenImageStarryNight
def OpenImageStarryNight(): 
    #Đọc ảnh: D:\opencv\sources\samples\python\tutorial_code\introduction\display_image
    #Hình ảnh lấy data: D:\opencv\sources\samples\data
    image = cv2.imread("D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/HinhAnh/OpenImageStarryNight/starry_night.jpg", cv2.IMREAD_COLOR)
    #cv2.imshow("Image_ThuHuongTest01", image)
    st.image(image,  width=600)

#Thiết lập hàm OpenCameraAndSaveFile 
def OpenCameraAndSaveFile():
    camera_device = 0 #Nếu có dường dẫn cho app android 'http://10.168../video'
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    #Dừng 2s khởi động camera 
    time.sleep(2)

    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        cv2.imshow("Image_ThuHuongTest02", frame) 

        #Nhấn phím ESC để lưu 
        #Ma ACSII cua phím ESC là 27
        key = cv2.waitKey(30) 
        if key == 27:
            break
        elif key == ord('S') or key == ord('s'):
            #Lấy ngày giờ hiện tại 
            timenow = time.localtime()

            #Đặt tên file lấy ngày giờ hiện tại 
            file_name = 'image_%04d_%02d_%02d_%02d_%02d_%02d.jpg' % (timenow.tm_year, timenow.tm_mon, timenow.tm_mday, timenow.tm_hour, timenow.tm_min, timenow.tm_sec)

            #Lưu file 
            cv2.imwrite(file_name, frame)

#Thiết lập hàm nhận diện gương mặt bằng mở camera
def NhanDienFaceOpenCamera(): 
    #Thư mục D:\opencv\sources\samples\python\tutorial_code\objectDetection\cascade_classifier
    #Nhận diện camera 
    def detectAndDisplay(frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

        cv2.imshow('Capture - Face detection', frame)

    face_cascade_name = 'D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_NhanDienFaceOpenCamera/haarcascade_frontalface_alt.xml'
    eyes_cascade_name = 'D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_NhanDienFaceOpenCamera/haarcascade_eye_tree_eyeglasses.xml'

    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()

    #-- 1. Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    camera_device = 0
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame)

        if cv2.waitKey(10) == 27:
            break

#Thiết lập hàm NhanDangKhuonMat_Facebook
def NhanDangKhuonMat_Facebook():
    #Thiết kế thông tin 3 button 
    with st.form("my_form"):
        btn_1 = st.form_submit_button("Bước 1: Lưu face cần thực hiện nhận diện")
        if btn_1:
            #Gọi hàm xử lý
            buoc1_faebook.main() 
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.write("Chúc mừng bạn hoàn thành xong bước 1")

        btn_2 = st.form_submit_button("Bước 2: Hệ thống tạo file training model xử lý - file tạo svc.pkl")
        if btn_2:
            #Gọi hàm xử lý
            buoc2_faebook.main() 
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.write("Chúc mừng bạn hoàn thành xong bước 2")
        btn_3 = st.form_submit_button("Bước 3: Nhận diện gương mặt")
        if btn_3:
            #Gọi hàm xử lý
            buoc3_faebook.main() 
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.write("Chúc mừng bạn hoàn thành xong bước 3")
  
#Thiết lập hàm face_detect_opencv_dnn
def Face_detect_opencv_dnn():
    #Hiển thị giao diện button
    with st.form("form1"):
        st.write("Button Face_detect_opencv_dnn")
        button = st.form_submit_button('Face_detect_opencv_dnn', disabled=False)
        if  button == True:
            #Gọi hàm xử lý 
            object_detection.main('face_detect_opencv_dnn')
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.text('Chúc mừng bạn hoàn thành tính năng Face_detect_opencv_dnn')
            
#Thiết lập hàm face_detect_opencv_dnn_caffe
def Face_detect_opencv_dnn_caffe():
    #Hiển thị giao diện button
    with st.form("form1"):
        st.write("Button Face_detect_opencv_dnn_caffe")
        button = st.form_submit_button('Face_detect_opencv_dnn_caffe', disabled=False)
        if  button == True:
            #Gọi hàm xử lý 
            object_detection.main('face_detect_opencv_dnn_caffe')
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.text('Chúc mừng bạn hoàn thành tính năng Face_detect_opencv_dnn_caffe')

#Thiết lập hàm Object_detect_opencv_yolo3
def Object_detect_opencv_yolo3():
    #Hiển thị giao diện button
    with st.form("form1"):
        st.write("Button Object_detect_opencv_yolo3")
        button = st.form_submit_button('Object_detect_opencv_yolo3', disabled=False)
        if  button == True:
            #Gọi hàm xử lý 
            object_detection.main('object_detect_opencv_yolo3')
            #Hủy cửa sổ Window trước khi thiết lập tính năng khác cho chương trình
            cv2.destroyAllWindows()
            #Thông báo giao diện hoàn thành xong tính năng 
            st.text('Chúc mừng bạn hoàn thành tính năng Object_detect_opencv_yolo3')
   
#Thiết lập hàm xử lý Upload File
def XuLyUploadFile(uploaded_files): 
    #Kiểm tra file đã được upload rồi mới thực hiện các bước tiếp theo
    if uploaded_files is not None:
        #Thiết lập menu thành phần khi cho file ảnh 
        menu_item = st.sidebar.selectbox('Mời chọn menu xử lý', ['Chương 3', 'Chương 4', 'Chương 9'])     
    
        #Kiểm tra nếu selectbox value là "Chương 3"
        if menu_item == 'Chương 3':
            #Thiết lập image_in
            filepath = 'D:\LV\BTThuHuong_HCMUTE\XuLyAnh\DoAnThuHuong\HinhAnh\Chuong3\\' + uploaded_files.name
            image_in = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            #Thiết lập menu phương thức xử lý ảnh
            chuong3_menu_function = st.sidebar.radio("Mời bạn chọn phương thức xử lý ảnh chương 3", 
                                                    ('Negative', 'Logarit', 'Power', 'PiecewiseLinear', 'Histogram', 'HistogramEqualization', 'LocalHistogram','HistogramStatistics',
                                                    'Smoothing', 'SmoothingGauss', 'MedianFilter', 'Sharpen', 'UnSharpMasking', 'Gradient'))

            #Kiểm tra radio là tính năng chương 3
            if chuong3_menu_function == 'Negative':
                XuLyAnh_Chuong3('Negative', image_in)
            elif chuong3_menu_function == 'Logarit':
                XuLyAnh_Chuong3('Logarit', image_in)
            elif chuong3_menu_function == 'Power':
                XuLyAnh_Chuong3('Power', image_in)
            elif chuong3_menu_function == 'PiecewiseLinear':
                XuLyAnh_Chuong3('PiecewiseLinear', image_in)
            elif chuong3_menu_function == 'Histogram':
                XuLyAnh_Chuong3('Histogram', image_in)
            elif chuong3_menu_function == 'HistogramEqualization':
                XuLyAnh_Chuong3('HistogramEqualization', image_in)
            elif chuong3_menu_function == 'LocalHistogram':
                XuLyAnh_Chuong3('LocalHistogram', image_in)
            elif chuong3_menu_function == 'HistogramStatistics':
                XuLyAnh_Chuong3('HistogramStatistics', image_in)
            elif chuong3_menu_function == 'Smoothing':
                XuLyAnh_Chuong3('Smoothing', image_in)
            elif chuong3_menu_function == 'SmoothingGauss':
                XuLyAnh_Chuong3('SmoothingGauss', image_in)
            elif chuong3_menu_function == 'MedianFilter':
                XuLyAnh_Chuong3('MedianFilter', image_in)
            elif chuong3_menu_function == 'Sharpen':
                XuLyAnh_Chuong3('Sharpen', image_in)
            elif chuong3_menu_function == 'UnSharpMasking':
                XuLyAnh_Chuong3('UnSharpMasking', image_in)
            elif chuong3_menu_function == 'Gradient':
                XuLyAnh_Chuong3('Gradient', image_in)

        elif menu_item == 'Chương 4':
            #Thiết lập image_in
            filepath = 'D:\LV\BTThuHuong_HCMUTE\XuLyAnh\DoAnThuHuong\HinhAnh\Chuong4\\' + uploaded_files.name
            image_in = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
            #Thiết lập menu phương thức xử lý ảnh
            chuong4_menu_function = st.sidebar.radio("Mời bạn chọn phương thức xử lý ảnh chương 4", 
                                                    ('Spectrum', 'FrequencyFilter', 'DrawFilter', 'RemoveMoire'))

            #Kiểm tra radio là tính năng chương 4
            if chuong4_menu_function == 'Spectrum':
                XuLyAnh_Chuong4('Spectrum', image_in)
            elif chuong4_menu_function == 'FrequencyFilter':
                XuLyAnh_Chuong4('FrequencyFilter', image_in)
            elif chuong4_menu_function == 'DrawFilter':
                XuLyAnh_Chuong4('DrawFilter', image_in)
            elif chuong4_menu_function == 'RemoveMoire':
                XuLyAnh_Chuong4('RemoveMoire', image_in)

        elif menu_item == 'Chương 9':
            #Thiết lập image_in
            filepath = 'D:\LV\BTThuHuong_HCMUTE\XuLyAnh\DoAnThuHuong\HinhAnh\Chuong9\\' + uploaded_files.name
            image_in = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            #Thiết lập menu phương thức xử lý ảnh
            chuong9_menu_function = st.sidebar.radio("Mời bạn chọn phương thức xử lý ảnh chương 9", 
                                                    ('Erosion', 'Dilation', 'OpeningClosing', 'Boundary', 'HoleFill', 'HoleFillMouse', 'MyConnectedComponent', 'ConnectedComponent', 'CountRice'))

            #Kiểm tra radio là tính năng chương 9
            if chuong9_menu_function == 'Erosion':
                XuLyAnh_Chuong9('Erosion', image_in)
            elif chuong9_menu_function == 'Dilation':
                XuLyAnh_Chuong9('Dilation', image_in)
            elif chuong9_menu_function == 'OpeningClosing':
                XuLyAnh_Chuong9('OpeningClosing', image_in)
            elif chuong9_menu_function == 'Boundary':
                XuLyAnh_Chuong9('Boundary', image_in)
            elif chuong9_menu_function == 'HoleFill':
                XuLyAnh_Chuong9('HoleFill', image_in)
            elif chuong9_menu_function == 'HoleFillMouse':
                XuLyAnh_Chuong9('HoleFillMouse', image_in)
            elif chuong9_menu_function == 'MyConnectedComponent':
                XuLyAnh_Chuong9('MyConnectedComponent', image_in)
            elif chuong9_menu_function == 'ConnectedComponent':
                XuLyAnh_Chuong9('ConnectedComponent', image_in)
            elif chuong9_menu_function == 'CountRice':
                XuLyAnh_Chuong9('CountRice', image_in)

    else : 
        cau1_Title = '<p style="font-family: Times New Roman; color: #6E2C00; font-size: 40px; font-weight: bold; text-align: center;">Mời bạn chọn hình ảnh</p>'
        st.markdown(cau1_Title, unsafe_allow_html=True)
        
#Kiểm tra thông tin menu page
if menu_page == '❄️ Mở hình ảnh starry_night':
    #Thiết lập nội dung tiêu đề
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Mở hình ảnh starry_night</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    OpenImageStarryNight()        
elif menu_page == '❄️ Mở máy ảnh và lưu file':
    #Thiết lập nội dung tiêu đề
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Mở máy ảnh và lưu file</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    OpenCameraAndSaveFile()
elif menu_page == '❄️ Nhận diện Face mở bằng Camera':
    #Thiết lập nội dung tiêu đề
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Nhận diện Face mở bằng Camera</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    NhanDienFaceOpenCamera()
elif menu_page == '❄️ Nhận diện face PP face detect opencv dnn':
    #Thiết lập nội dung tiêu đề 
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Nhận diện face PP face detect opencv dnn</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    Face_detect_opencv_dnn()
elif menu_page == '❄️ Nhận diện face PP face detect opencv dnn_caffe':
    #Thiết lập nội dung tiêu đề 
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Nhận diện face PP face detect opencv dnn_caffe</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    Face_detect_opencv_dnn_caffe()
elif menu_page == '❄️ Nhận diện object detect opencv yolo3':
    #Thiết lập nội dung tiêu đề 
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Nhận diện object detect opencv yolo3</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    Object_detect_opencv_yolo3()
elif menu_page == '❄️ Nhận diện NhanDangKhuonMat Facebook':
    #Thiết lập nội dung tiêu đề 
    menu_content = '<p style="font-family: Times New Roman; color: #1E8449; font-size: 40px; font-weight: bold; text-align: center;">Nhận diện NhanDangKhuonMat Facebook</p>'
    st.markdown(menu_content, unsafe_allow_html=True)
    #Gọi hàm thực thi xử lý tính năng 
    NhanDangKhuonMat_Facebook()
elif menu_page == '❄️ Chọn thuật toán xử lý ảnh': 
    #Chọn thông tin file image
    uploaded_files = st.sidebar.file_uploader("Chọn file image", type=['csv', 'png', 'tif', 'jpg'])
    #Gọi hàm thực thi xử lý tính năng 
    XuLyUploadFile(uploaded_files)
