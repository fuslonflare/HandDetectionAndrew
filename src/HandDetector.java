import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.*;

import java.awt.*;
import java.awt.Point;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Created by Phuwarin on 1/29/2017.
 */
public class HandDetector {
    private static final int IMG_SCALE = 2;  // scaling applied to webcam image

    private static final float SMALLEST_AREA = 600.0f;    // was 100.0f;
    // ignore smaller contour areas

    private static final int MAX_POINTS = 20;   // max number of points stored in an array

    // used for simiplifying the defects list
    private static final int MIN_FINGER_DEPTH = 20;
    private static final int MAX_FINGER_ANGLE = 60;   // degrees

    // angle ranges of thumb and index finger of the left hand relative to its COG
    private static final int MIN_THUMB = 120;
    private static final int MAX_THUMB = 200;

    private static final int MIN_INDEX = 60;
    private static final int MAX_INDEX = 120;


    // HSV ranges defining the glove colour
    private int hueLower, hueUpper, satLower, satUpper, briLower, briUpper;

    // JavaCV elements
    private IplImage scaleImg;     // for resizing the webcam image
    private IplImage hsvImg;       // HSV version of webcam image
    private IplImage imgThreshed;  // threshold for HSV settings
    private CvMemStorage contourStorage, approxStorage, hullStorage, defectsStorage;

    private Font msgFont;

    // hand details
    private Point cogPt;           // center of gravity (COG) of contour
    private int contourAxisAngle;
    // contour's main axis angle relative to the horizontal (in degrees)

    // defects data for the hand contour
    private Point[] tipPts, foldPts;
    private float[] depths;
    private ArrayList<Point> fingerTips;

    // finger identifications
    private ArrayList<FingerName> namedFingers;

    public HandDetector(String hsvFnm, int width, int height) {
        scaleImg = IplImage.create(width / IMG_SCALE, height / IMG_SCALE, 8, 3);
        hsvImg = IplImage.create(width / IMG_SCALE, height / IMG_SCALE, 8, 3);     // for the HSV image
        imgThreshed = IplImage.create(width / IMG_SCALE, height / IMG_SCALE, 8, 1);   // threshold image

        // storage for contour, hull, and defect calculations by OpenCV
        contourStorage = CvMemStorage.create();
        approxStorage = CvMemStorage.create();
        hullStorage = CvMemStorage.create();
        defectsStorage = CvMemStorage.create();

        msgFont = new Font("SansSerif", Font.BOLD, 18);

        cogPt = new Point();
        fingerTips = new ArrayList<Point>();
        namedFingers = new ArrayList<FingerName>();

        tipPts = new Point[MAX_POINTS];   // coords of the finger tips
        foldPts = new Point[MAX_POINTS];  // coords of the skin folds between fingers
        depths = new float[MAX_POINTS];   // distances from tips to folds

        setHSVRanges(hsvFnm);
    }  // end of defaultConstructor()

    private void setHSVRanges(String fnm) {
        /* read in three lines to set the lower/upper HSV ranges for the user's glove.
        These were previously stored using the HSV Selector application */
        try {
            BufferedReader in = new BufferedReader(new FileReader(fnm));
            String line = in.readLine();   // get hues
            String[] toks = line.split("\\s+");
            hueLower = Integer.parseInt(toks[1]);
            hueUpper = Integer.parseInt(toks[2]);

            line = in.readLine();   // get saturations
            toks = line.split("\\s+");
            satLower = Integer.parseInt(toks[1]);
            satUpper = Integer.parseInt(toks[2]);

            line = in.readLine();   // get brightnesses
            toks = line.split("\\s+");
            briLower = Integer.parseInt(toks[1]);
            briUpper = Integer.parseInt(toks[2]);

            in.close();
            System.out.println("Read HSV ranges from " + fnm);
        } catch (Exception e) {
            System.out.println("Could not read HSV ranges from " + fnm);
            System.exit(1);
        }
    }  // end of setHSVRanges()

    public void update(IplImage im) {
        cvResize(im, scaleImg); // reduce the size of the image to make processing faster
        cvCvtColor(scaleImg, hsvImg, CV_BGR2HSV); // convert image format to HSV
        cvInRangeS(hsvImg, cvScalar(hueLower, satLower, briLower, 0.0),
                cvScalar(hueUpper, satUpper, briUpper, 0.0),
                imgThreshed); // threshold image using loaded HSV settings for user's glove
        cvMorphologyEx(imgThreshed, imgThreshed, null, null, CV_MOP_OPEN, 1);
            /* erosion followed by dilation on the image to remove
            specks of white while retaining the image size
             */
        CvSeq bigContour = findBiggestContour(imgThreshed);
        if (bigContour == null) {
            return;
        }
        extractContourInfo(bigContour, IMG_SCALE); // find the COG and angle to horizontal of the contour
        findFingerTips(bigContour, IMG_SCALE); // detect the fingertips position in the contour
        nameFingers(cogPt, contourAxisAngle, fingerTips); // end of update()
    }

    private CvSeq findBiggestContour(IplImage imgThreshed) {
        CvSeq bigContour = null;

        // generate all the contours in the threshold image as a list
        CvSeq contours = new CvSeq(null);
        cvFindContours(imgThreshed, contourStorage, contours,
                Loader.sizeof(CvContour.class),
                CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        // find the largest contour in the list based on bounded box size
        float maxArea = SMALLEST_AREA;
        CvBox2D maxBox = null;
        while (!contours.isNull()) {
            if (contours.elem_size() > 0) {
                CvBox2D box = cvMinAreaRect2(contours, contourStorage);
                if (!box.isNull()) {
                    CvSize2D32f size = box.size();
                    float area = size.width() * size.height();
                    if (area > maxArea) {
                        maxArea = area;
                        bigContour = contours;
                    }
                }
            }
            contours = contours.h_next();
        }
        return bigContour;
    } // end findBiggestContour()


    private void extractContourInfo(CvSeq bigContour,
                                    int scale) {
        CvMoments moments = new CvMoments();
        cvMoments(bigContour, moments, 1);

        // center of gravity
        double m00 = cvGetSpatialMoment(moments, 0, 0);
        double m10 = cvGetSpatialMoment(moments, 1, 0);
        double m01 = cvGetSpatialMoment(moments, 0, 1);

        if (m00 != 0.0) { // calculate center
            int xCenter = (int) (Math.round(m10 / m00) * scale);
            int yCenter = (int) (Math.round(m01 / m00) * scale);
            cogPt.setLocation(xCenter, yCenter);
        }

        double m11 = cvGetCentralMoment(moments, 1, 1);
        double m20 = cvGetCentralMoment(moments, 2, 0);
        double m02 = cvGetCentralMoment(moments, 0, 2);
        contourAxisAngle = calculateTilt(m11, m20, m02);

        // deal with hand contour pointing downwards
        /* uses fingertips information generated on the last update of
        the hand, so will be ood (out-of-date)
        * */

        if (fingerTips.size() > 0) {
            int yTotal = 0;
            for (Point aPoint : fingerTips) {
                yTotal += aPoint.y;
            }
            int avgYFinger = yTotal / fingerTips.size();
            if (avgYFinger > cogPt.y) { // finger below cog
                contourAxisAngle += 180;
            }
        }
        contourAxisAngle = 180 - contourAxisAngle;
        /* this makes the angle relative to a positive y-axis that
        runs up the screen */
    } // end of extractContourInfo

    private int calculateTilt(double m11,
                              double m20,
                              double m02) {
        double diff = m20 - m02;
        if (diff == 0) {
            if (m11 == 0) {
                return 0;
            } else if (m11 > 0) {
                return 45;
            } else { // m11 < 0
                return -45;
            }
        }

        double theta = 0.5 * Math.atan2(2 * m11, diff);
        int tilt = (int) Math.round(Math.toDegrees(theta));

        if ((diff > 0) && (m11 == 0)) {
            return 0;
        } else if ((diff < 0) && (m11 == 0)) {
            return -90;
        } else if ((diff > 0) && (m11 > 0)) { // 0~45 degree
            return tilt;
        } else if ((diff > 0) && (m11 < 0)) { // -45~0 degree
            return (180 + tilt); // change to CC (counter-clockwise) angle
        } else if ((diff < 0) && (m11 > 0)) { // 45~90 degree
            return tilt;
        } else if ((diff < 0) && (m11 < 0)) { // -90~-45 degree
            return (180 + tilt); // change to counter-clockwise angle
        }

        System.err.print("Error in moments for tilt angle");
        return 0;
    } // end of calculationTilt()

    private void findFingerTips(CvSeq bigContour,
                                int scale) {
        CvSeq approxContour = cvApproxPoly(
                bigContour, Loader.sizeof(CvContour.class),
                approxStorage, CV_POLY_APPROX_DP,
                3, 1);
        // reduce number of points in the contour

        CvSeq hullSeq = cvConvexHull2(
                approxContour, hullStorage, CV_COUNTER_CLOCKWISE, 0);
        // find the convex hull around the contour

        CvSeq defects = cvConvexityDefects(
                approxContour, hullSeq, defectsStorage);
        // find the defect difference between the contour and hull

        int defectsTotal = defects.total();
        if (defectsTotal > MAX_POINTS) {
            System.out.println("Processing " + MAX_POINTS + " defect pts");
            defectsTotal = MAX_POINTS;
        }

        // copy defect information from defects sequence into arrays
        for (int i = 0; i < defectsTotal; i++) {
            Pointer pointer = cvGetSeqElem(defects, i);
            CvConvexityDefect cdf = new CvConvexityDefect(pointer);

            CvPoint startPt = cdf.start();
            tipPts[i] = new Point(Math.round(startPt.x() * scale),
                    Math.round(startPt.y() * scale));
            // array contains coordinates of the fingertips

            CvPoint endPt = cdf.end();
            CvPoint depthPt = cdf.depth_point();
            foldPts[i] = new Point(Math.round(depthPt.x() * scale),
                    Math.round(depthPt.y() * scale));
            // array contains coordinates of the skin fold between fingers

            depths[i] = cdf.depth() * scale;
            // array contains distances from tips to folds
        }
        reduceTips(defectsTotal, tipPts, foldPts, depths);
    } // end of findFingerTips()

    private void reduceTips(int numPoints,
                            Point[] tipPts,
                            Point[] foldPts,
                            float[] depths) {
        fingerTips.clear();

        for (int i = 0; i < numPoints; i++) {
            if (depths[i] < MIN_FINGER_DEPTH) {
                continue;
            }

            // look at fold points on either side of a trip
            int pdx = (i == 0) ? (numPoints - 1) : (i - 1); // predecessor of i
            int sdx = (i == numPoints - 1) ? 0 : (i + 1); // successor of i

            int angle = angleBetween(tipPts[i], foldPts[pdx], foldPts[sdx]);
            if (angle >= MAX_FINGER_ANGLE) {
                continue; // angle between finger and folds too wide
            }

            // this point is probably a fingertips, so add to list
            fingerTips.add(tipPts[i]);
        }
    } // end of reduceTips()

    private int angleBetween(Point tip,
                             Point next,
                             Point prev) {
        // calculate the angle between the tip and it's neighboring folds
        // in integer degree
        return (int) Math.abs(Math.round(
                Math.toDegrees(Math.atan2(next.x - tip.x, next.y - tip.y) -
                        Math.atan2(prev.x - tip.x, prev.y - tip.y))));
    }


    private void nameFingers(Point cogPt,
                             int contourAxisAngle,
                             ArrayList<Point> fingerTips) {
        // reset all named fingers to unknown
        namedFingers.clear();

        for (int i = 0; i < fingerTips.size(); i++) {
            namedFingers.add(FingerName.UNKNOWN);
        }
        labelThumbIndex(fingerTips, namedFingers);
        labelUnknowns(namedFingers);
    } // end of nameFingers()

    private void labelThumbIndex(ArrayList<Point> fingerTips,
                                 ArrayList<FingerName> nms) {
        boolean foundThumb = false;
        boolean foundIndex = false;
        int i = fingerTips.size() - 1;
        while (i >= 0) {
            int angle = angleToCOG(fingerTips.get(i),
                    cogPt,
                    contourAxisAngle);
            // check for thumb
            if ((angle <= MAX_THUMB) && (angle > MIN_THUMB) && !foundThumb) {
                nms.set(i, FingerName.THUMB);
                foundThumb = true;
            }

            // check for index
            if ((angle <= MAX_INDEX) && (angle > MIN_INDEX) && !foundIndex) {
                nms.set(i, FingerName.INDEX);
                foundIndex = true;
            }
            i--;
        }
    } // end of labelThumbIndex

    private int angleToCOG(Point tipPt,
                           Point cogPt,
                           int contourAxisAngle) {
        int yOffset = cogPt.y - tipPt.y;  // make y positive up screen
        int xOffset = tipPt.x - cogPt.x;
        double theta = Math.atan2(yOffset, xOffset);
        int angleTip = (int) Math.round(Math.toDegrees(theta));
        return angleTip + (90 - contourAxisAngle);
        // this ensures that the hand is orientated straight up
    } // end of angleToCOG()


    private void labelUnknowns(ArrayList<FingerName> nms) {
        // find first named finger
        int i = 0;
        while ((i < nms.size()) && (nms.get(i) == FingerName.UNKNOWN)) {
            i++;
        }
        if (i == nms.size()) {  // no named fingers found, so give up
            return;
        }

        FingerName name = nms.get(i);
        labelPrev(nms, i, name);    // fill-in backwards
        labelFwd(nms, i, name);    // fill-in forwards
    }  // end of labelUnknowns()

    private void labelPrev(ArrayList<FingerName> nms,
                           int i,
                           FingerName name) {
        // move backwards through fingers list labelling unknown fingers
        i--;

        while ((i >= 0) && (name != FingerName.UNKNOWN)) {
            if (nms.get(i) == FingerName.UNKNOWN) { // unknown finger
                name = name.getPrev();
                if (!usedName(nms, name)) {
                    nms.set(i, name);
                }
            } else {   // finger is named already
                name = nms.get(i);
            }
            i--;
        }
    }  // end of labelPrev()

    private boolean usedName(ArrayList<FingerName> nms, FingerName name) {
        // does the fingers list contain name already?

        for (FingerName fn : nms) {
            if (fn == name) {
                return true;
            }
        }
        return false;
    }  // end of usedName()


    private void labelFwd(ArrayList<FingerName> nms, int i, FingerName name) {
        // move forward through fingers list labelling unknown fingers
        i++;

        while ((i < nms.size()) && (name != FingerName.UNKNOWN)) {
            if (nms.get(i) == FingerName.UNKNOWN) {  // unknown finger
                name = name.getNext();
                if (!usedName(nms, name)) {
                    nms.set(i, name);
                }
            } else {    // finger is named already
                name = nms.get(i);
            }
            i++;
        }
    } // end of labelFwd()

    public void draw(Graphics2D g2d) {
        // draw information about the finger tips and the hand COG
        if (fingerTips.size() == 0) {
            return;
        }

        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);  // line smoothing
        g2d.setPaint(Color.YELLOW);
        g2d.setStroke(new BasicStroke(4));  // thick yellow pen

        // label the finger tips in red or green, and draw COG lines to named tips
        g2d.setFont(msgFont);
        for (int i = 0; i < fingerTips.size(); i++) {
            Point pt = fingerTips.get(i);
            if (namedFingers.get(i) == FingerName.UNKNOWN) {
                g2d.setPaint(Color.RED);   // unnamed finger tip is red
                g2d.drawOval(pt.x - 8, pt.y - 8, 16, 16);
                g2d.drawString("" + i, pt.x, pt.y - 10);   // label it with a digit
            } else {   // draw yellow line to the named finger tip from COG
                g2d.setPaint(Color.YELLOW);
                g2d.drawLine(cogPt.x, cogPt.y, pt.x, pt.y);

                g2d.setPaint(Color.GREEN);   // named finger tip is green
                g2d.drawOval(pt.x - 8, pt.y - 8, 16, 16);
                g2d.drawString(namedFingers.get(i).toString().toLowerCase(), pt.x, pt.y - 10);
            }
        }

        // draw COG
        g2d.setPaint(Color.GREEN);
        g2d.fillOval(cogPt.x - 8, cogPt.y - 8, 16, 16);
    }  // end of draw()
}
