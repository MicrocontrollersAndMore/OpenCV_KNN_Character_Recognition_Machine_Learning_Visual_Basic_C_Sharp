// GenerateData.cs

// using Emgu CV 2.4.10

// add the following components to your form:
// btnOpenTrainingImage (Button)
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
namespace GenerateData4 {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    public partial class frmMain : Form {

        // module level variables /////////////////////////////////////////////////////////////////
        const int MIN_CONTOUR_AREA = 100;

        const int RESIZED_IMAGE_WIDTH = 20;
        const int RESIZED_IMAGE_HEIGHT = 30;

        int intNumberOfTrainingSamples;

        // constructor ////////////////////////////////////////////////////////////////////////////
        public frmMain() {
            InitializeComponent();
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        private void btnOpenTrainingImage_Click(object sender, EventArgs e) {
            DialogResult drChosenFile;

            drChosenFile = ofdOpenFile.ShowDialog();                // open file dialog

            if (drChosenFile != DialogResult.OK || ofdOpenFile.FileName == "") {        // if user chose Cancel or filename is blank . . .
                lblChosenFile.Text = "file not chosen";             // show error message on label
                return;                                             // and exit function
            }

            Image<Bgr, Byte> imgTrainingNumbers;                // this is the main input image

            try {
                imgTrainingNumbers = new Image<Bgr, Byte>(ofdOpenFile.FileName);        // open image
            } catch (Exception ex) {                                                    // if error occurred
                lblChosenFile.Text = "unable to open image, error: " + ex.Message;      // show error message on label
                return;                                                                 // and exit function
            }

            if (imgTrainingNumbers == null) {                               // if image could not be opened
                lblChosenFile.Text = "unable to open image";                // show error message on label
                return;                                                     // and exit function
            }

            lblChosenFile.Text = ofdOpenFile.FileName;          // update label with file name

            Image<Gray, Byte> imgGrayscale;                 //
            Image<Gray, Byte> imgBlurred;                   //
            Image<Gray, Byte> imgThresh;                    // declare various images
            Image<Gray, Byte> imgThreshCopy;                //
            Image<Gray, Byte> imgContours;                  //

            Contour<Point> contours;

                                            // possible chars we are interested in are digits 0 through 9, put these in list intValidChars
            List<int> intValidChars = new List<int> { (int)'0', (int)'1', (int)'2', (int)'3', (int)'4', (int)'5', (int)'6', (int)'7', (int)'8', (int)'9' };

            imgGrayscale = imgTrainingNumbers.Convert<Gray, Byte>();        // convert to grayscale

            imgBlurred = imgGrayscale.SmoothGaussian(5);                // blur

                                                // filter image from grayscale to black and white
            imgThresh = imgBlurred.ThresholdAdaptive(new Gray(255), ADAPTIVE_THRESHOLD_TYPE.CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH.CV_THRESH_BINARY_INV, 11, new Gray(2));

            CvInvoke.cvShowImage("imgThresh", imgThresh);               // show threshold image for reference

            imgThreshCopy = imgThresh.Clone();                  // make a copy of the thresh image, this in necessary b/c findContours modifies the image

                                            // get external countours only
            contours = imgThreshCopy.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_EXTERNAL);

                                                // next we count the contours
            intNumberOfTrainingSamples = 0;         // init number of contours (i.e. training samples) to zero

            while (contours != null) {
                intNumberOfTrainingSamples = intNumberOfTrainingSamples + 1;
                contours = contours.HNext;
            }

            contours = imgThreshCopy.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_EXTERNAL);      // get contours again to go back to beginning

            imgContours = new Image<Gray, Byte>(imgThresh.Size);        // instantiate contours image

                                                                        // draw contours onto contours image
            CvInvoke.cvDrawContours(imgContours, contours, new MCvScalar(255), new MCvScalar(255), 100, 1, LINE_TYPE.CV_AA, new Point(0, 0));

            CvInvoke.cvShowImage("imgContours", imgContours);           // show contours image for reference

                                            // this is our classifications data structure
            Matrix<Single> mtxClassifications = new Matrix<Single>(intNumberOfTrainingSamples, 1);

                                            // this is our training images data structure, note we will have to perform some conversions to write to this later
            Matrix<Single> mtxTrainingImages = new Matrix<Single>(intNumberOfTrainingSamples, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT);

                                                    // this keeps track of which row we are on in both classifications and training images,
            int intTrainingDataRowToAdd = 0;        // note that each sample will correspond to one row in
                                                    // both the classifications XML file and the training images XML file

            while (contours != null) {
                Contour<Point> contour = contours.ApproxPoly(contours.Perimeter * 0.0001);          //get the current contour, note that the lower the multiplier, the higher the precision
                if (ContourIsValid(contour)) {                                  // if contour is big enough to consider
                    Rectangle rect = contour.BoundingRectangle;                 // get the bounding rect
                    imgTrainingNumbers.Draw(rect, new Bgr(Color.Red), 2);       // draw red rectangle around each contour as we ask user for input
                    Image<Gray, Byte> imgROI = imgThresh.Copy(rect);            // get ROI image of current char

                                        // resize image, this is necessary for recognition and storage
                    Image<Gray, Byte> imgROIResized = imgROI.Resize(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT, INTER.CV_INTER_LINEAR);

                    CvInvoke.cvShowImage("imgROI", imgROI);                                 // show ROI image for reference
                    CvInvoke.cvShowImage("imgROIResized", imgROIResized);                   // show resized ROI image for reference
                    CvInvoke.cvShowImage("imgTrainingNumbers", imgTrainingNumbers);         // show training numbers image, this will now have red rectangles drawn on it

                    int intChar = CvInvoke.cvWaitKey(0);        // get key press

                    if (intChar == 27) {                // if esc key was pressed
                        return;                         // exit the function
                    } else if (intValidChars.Contains(intChar)) {           // else if the char is in the list of chars we are looking for . . .
                        
                        mtxClassifications[intTrainingDataRowToAdd, 0] = Convert.ToSingle(intChar);         // write classification char to classifications Matrix

                        // now add the training image (some conversion is necessary first) . . .
                        // note that we have to covert the images to Matrix(Of Single) type, this is necessary to pass into the KNearest object call to train
                        Matrix<Single> mtxTemp = new Matrix<Single>(imgROIResized.Size);                    // declare a Matrix of the same dimensions as the Image we are adding to the data structure of training images
                        Matrix<Single> mtxTempReshaped = new Matrix<Single>(1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT);         // declare a flattened (only 1 row) matrix of the same total size

                        CvInvoke.cvConvert(imgROIResized, mtxTemp);             // convert Image to a Matrix of Singles with the same dimensions

                        for (int intRow = 0; intRow < RESIZED_IMAGE_HEIGHT; intRow++) {         // flatten Matrix into one row by RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT number of columns
                            for (int intCol = 0; intCol < RESIZED_IMAGE_WIDTH; intCol++) {
                                mtxTempReshaped[0, (intRow * RESIZED_IMAGE_WIDTH) + intCol] = mtxTemp[intRow, intCol];
                            }
                        }

                        for (int intCol = 0; intCol < RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT; intCol++) {       // write flattened Matrix into one row of training images Matrix
                            mtxTrainingImages[intTrainingDataRowToAdd, intCol] = mtxTempReshaped[0, intCol];
                        }

                        intTrainingDataRowToAdd = intTrainingDataRowToAdd + 1;          // increment which row, i.e. sample we are on
                    }   // end else if
                }   // end if
                contours = contours.HNext;              // move on to next contour
            }   // end while

            txtInfo.Text = txtInfo.Text + "training complete !!" + Environment.NewLine + Environment.NewLine;

                    // save classifications to file /////////////////////////////////////

            XmlSerializer xmlSerializer = new XmlSerializer(mtxClassifications.GetType());
            StreamWriter streamWriter;

            try {
                streamWriter = new StreamWriter("classifications.xml");                     // attempt to open classifications file
            } catch (Exception ex) {                                                        // if error is encountered, show error and return
                txtInfo.Text = Environment.NewLine + txtInfo.Text + "unable to open 'classifications.xml', error:" + Environment.NewLine;
                txtInfo.Text = txtInfo.Text + ex.Message + Environment.NewLine + Environment.NewLine;
                return;
            }

            xmlSerializer.Serialize(streamWriter, mtxClassifications);
            streamWriter.Close();

                    // save training images to file /////////////////////////////////////

            xmlSerializer = new XmlSerializer(mtxTrainingImages.GetType());
            
            try {
                streamWriter = new StreamWriter("images.xml");                     // attempt to open classifications file
            } catch (Exception ex) {                                                        // if error is encountered, show error and return
                txtInfo.Text = Environment.NewLine + txtInfo.Text + "unable to open 'images.xml', error:" + Environment.NewLine;
                txtInfo.Text = txtInfo.Text + ex.Message + Environment.NewLine + Environment.NewLine;
                return;
            }

            xmlSerializer.Serialize(streamWriter, mtxTrainingImages);
            streamWriter.Close();

            txtInfo.Text = Environment.NewLine + txtInfo.Text + "file writing done" + Environment.NewLine;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        bool ContourIsValid(Contour<Point> contour) {
            if (contour.Area >= MIN_CONTOUR_AREA) {         // obviously in a production grade program
                return true;                                // we would have a much more robust function for
            }                                               // identifying if a contour is valid !!
            return false;
        }

    }   // end class

}   // end namespace
