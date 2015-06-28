// TrainAndTest.cs

// using Emgu CV 2.4.10

// add the following components to your form:
// btnOpenTestImage (Button)
// lblChosenFile (Label)
// txtInfo (TextBox)
// ofdOpenFile (OpenFileDialog)

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;                      //
using Emgu.CV.CvEnum;               // Emgu Cv imports
using Emgu.CV.Structure;            //
using Emgu.CV.UI;                   //
using Emgu.CV.ML;                   //

using System.Xml;                   //
using System.Xml.Serialization;     // these imports are for writing Matrix objects to file, see end of program
using System.IO;                    //

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace TrainAndTest4 {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    public partial class frmMain : Form {

        // member variables ///////////////////////////////////////////////////////////////////////
        const int MIN_CONTOUR_AREA = 100;

        const int RESIZED_IMAGE_WIDTH = 20;
        const int RESIZED_IMAGE_HEIGHT = 30;

        int intNumberOfTrainingSamples;

        // constructor ////////////////////////////////////////////////////////////////////////////
        public frmMain() {
            InitializeComponent();
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        private void btnOpenTestImage_Click(object sender, EventArgs e) {
                    // note: we effectively have to read the first XML file twice
                    // first, we read the file to get the number of rows (which is the same as the number of samples).
                    // the first time reading the file we can't get the data yet, since we don't know how many rows of data there are
                    // next, reinstantiate our classifications Matrix and training images Matrix with the correct number of rows
                    // then, read the file again and this time read the data into our resized classifications Matrix and training images Matrix

            Matrix<Single> mtxClassifications = new Matrix<Single>(1, 1);       // for the first time through, declare these to be 1 row by 1 column
            Matrix<Single> mtxTrainingImages = new Matrix<Single>(1, 1);        // we will resize these when we know the number of rows (i.e. number of training samples)

                    // possible chars we are interested in are digits 0 through 9
            List<int> intValidChars = new List<int> { (int)'0', (int)'1', (int)'2', (int)'3', (int)'4', (int)'5', (int)'6', (int)'7', (int)'8', (int)'9' };

            XmlSerializer xmlSerializer = new XmlSerializer(mtxClassifications.GetType());      // these variables are for
            StreamReader streamReader;                                                          // reading from the XML files

            try {
                streamReader = new StreamReader("classifications.xml");                     // attempt to open classifications file
            } catch(Exception ex) {                                                         // if error is encountered, show error and return
                txtInfo.Text = Environment.NewLine + txtInfo.Text + "unable to open 'classifications.xml', error:" + Environment.NewLine;
                txtInfo.Text = txtInfo.Text + ex.Message + Environment.NewLine + Environment.NewLine;
                return;
            }

                    // read from the classifications file the 1st time, this is only to get the number of rows, not the actual data
            mtxClassifications = (Matrix<Single>)xmlSerializer.Deserialize(streamReader);
            
            streamReader.Close();               // close the classifications XML file

            intNumberOfTrainingSamples = mtxClassifications.Rows;       // get the number of rows, i.e. the number of training samples

                    // now that we know the number of rows, reinstantiate classifications Matrix and training images Matrix with the actual number of rows
            mtxClassifications = new Matrix<Single>(intNumberOfTrainingSamples, 1);
            mtxTrainingImages = new Matrix<Single>(intNumberOfTrainingSamples, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT);

            try {
                streamReader = new StreamReader("classifications.xml");                 // attempt to reinitialize the stream reader
            } catch (Exception ex) {                                                    // if error is encountered, show error and return
                txtInfo.Text = Environment.NewLine + txtInfo.Text + "unable to open 'classifications.xml', error:" + Environment.NewLine;
                txtInfo.Text = txtInfo.Text + ex.Message + Environment.NewLine + Environment.NewLine;
                return;
            }

                    // read from the classifications file again, this time we can get the actual data
            mtxClassifications = (Matrix<Single>)xmlSerializer.Deserialize(streamReader);

            streamReader.Close();           // close the classifications XML file

            xmlSerializer = new XmlSerializer(mtxTrainingImages.GetType());     // reinstantiate file reading variables

            try {
                streamReader = new StreamReader("images.xml");
            } catch (Exception ex) {                                            // if error is encountered, show error and return
                txtInfo.Text = Environment.NewLine + txtInfo.Text + "unable to open 'images.xml', error:" + Environment.NewLine;
                txtInfo.Text = txtInfo.Text + ex.Message + Environment.NewLine + Environment.NewLine;
                return;
            }

            mtxTrainingImages = (Matrix<Single>)xmlSerializer.Deserialize(streamReader);        // read from training images file
            streamReader.Close();           // close the training images XML file

                    // train //////////////////////////////////////////////////////////

            KNearest kNearest = new KNearest();                                                 // instantiate KNN object
            kNearest.Train(mtxTrainingImages, mtxClassifications, null, false, 1,false);        // call to train

                    // test ///////////////////////////////////////////////////////////////////////

            DialogResult drChosenFile;

            drChosenFile = ofdOpenFile.ShowDialog();            // open file dialog

            if (drChosenFile != DialogResult.OK || ofdOpenFile.FileName == "") {            // if user chose Cancel or filename is blank . . .
                lblChosenFile.Text = "file not chosen";         // show error message on label
                return;                                         // and exit function
            }

            Image<Bgr, Byte> imgTestingNumbers;                 // this is the main input image

            try {
                imgTestingNumbers = new Image<Bgr, Byte>(ofdOpenFile.FileName);         // open image
            } catch(Exception ex) {                                                     // if error occurred
                lblChosenFile.Text = "unable to open image, error: " + ex.Message;      // show error message on label
                return;                                                                 // and exit function
            }

            if(imgTestingNumbers == null) {                         //if image could not be opened
                lblChosenFile.Text = "unable to open image";        // show error message on label
                return;                                             // and exit function
            }

            lblChosenFile.Text = ofdOpenFile.FileName;              // update label with file name

            Image<Gray, Byte> imgGrayscale;              //
            Image<Gray, Byte> imgBlurred;                // declare various images
            Image<Gray, Byte> imgThresh;                 //
            Image<Gray, Byte> imgThreshCopy;             //

            Contour<Point> contours;

            imgGrayscale = imgTestingNumbers.Convert<Gray, Byte>();         // convert to grayscale

            imgBlurred = imgGrayscale.SmoothGaussian(5);                    // blur

                                        // filter image from grayscale to black and white
            imgThresh = imgBlurred.ThresholdAdaptive(new Gray(255), ADAPTIVE_THRESHOLD_TYPE.CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH.CV_THRESH_BINARY_INV, 11, new Gray(2));

            imgThreshCopy = imgThresh.Clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the image

                                        // get external countours only
            contours = imgThreshCopy.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_EXTERNAL);

            List<Contour<Point> > listOfContours = new List<Contour<Point> >();             // declare a list of contours and a list of valid contours,
            List<Contour<Point> > listOfValidContours = new List<Contour<Point> >();        // this is necessary for removing invalid contours and sorting from left to right

                                        // populate list of contours
            while(contours != null) {               // for each contour
                Contour<Point> contour = contours.ApproxPoly(contours.Perimeter * 0.0001);      // get the current contour, note that the lower the multiplier, the higher the precision
                listOfContours.Add(contour);                                                    // add to list of contours
                contours = contours.HNext;                                                      // move on to next contour
            }
                                        // this next loop removes the invalid contours
            foreach (Contour<Point> contour in listOfContours) {// for each contour
                if(ContourIsValid(contour)) {// if contour is valid
                    listOfValidContours.Add(contour);// add to list of valid contours
                }
            }

                        // sort contours from left to right
            listOfValidContours.Sort((oneContour, otherContour) => oneContour.BoundingRectangle.X.CompareTo(otherContour.BoundingRectangle.X));

            String strFinalString = "";             // declare final string, this will have the final number sequence by the end of the program

            foreach (Contour<Point> contour in listOfValidContours) {       // for each contour in list of valid contours
                Rectangle rect = contour.BoundingRectangle;                     // get the bounding rect
                imgTestingNumbers.Draw(rect, new Bgr(Color.Green), 2);          // draw green rect around the current char
                Image<Gray, Byte> imgROI = imgThresh.Copy(rect);                // get ROI image of bounding rect

                                        // resize image, this is necessary for recognition
                Image<Gray, Byte> imgROIResized = imgROI.Resize(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT, INTER.CV_INTER_LINEAR);

                Matrix<Single> mtxTemp = new Matrix<Single>(imgROIResized.Size);                                        // declare a Matrix of the same dimensions as the Image we are adding to the data structure of training images
                Matrix<Single> mtxTempReshaped = new Matrix<Single>(1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT);     // declare a flattened (only 1 row) matrix of the same total size

                CvInvoke.cvConvert(imgROIResized, mtxTemp);             // convert Image to a Matrix of Singles with the same dimensions

                for(int intRow = 0; intRow < RESIZED_IMAGE_HEIGHT; intRow++){       // flatten Matrix into one row by RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT number of columns
                    for(int intCol = 0; intCol < RESIZED_IMAGE_WIDTH; intCol++) {
                        mtxTempReshaped[0, (intRow * RESIZED_IMAGE_WIDTH) + intCol] = mtxTemp[intRow, intCol];
                    }
                }

                Single sngCurrentChar = kNearest.FindNearest(mtxTempReshaped, 1, null, null, null, null);       // finally we can call find_nearest !!!

                strFinalString = strFinalString + Convert.ToChar(Convert.ToInt32(sngCurrentChar));              // append current char to full string
            }   // end foreach

                        // show the full string
            txtInfo.Text = Environment.NewLine + Environment.NewLine + txtInfo.Text + "number read from image = " + strFinalString + Environment.NewLine;

            CvInvoke.cvShowImage("imgTestingNumbers", imgTestingNumbers);       // show input image with green boxes drawn around found digits
        }   // end function

        ///////////////////////////////////////////////////////////////////////////////////////////
        bool ContourIsValid(Contour<Point> contour) {
            if (contour.Area >= MIN_CONTOUR_AREA) {         // obviously in a production grade program
                return true;                                // we would have a much more robust function for
            }                                               // identifying if a contour is valid !!
            return false;
        }

    }   // end class

}   // end namespace
