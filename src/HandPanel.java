/**
 * Created by Phuwarin on 1/29/2017.
 */

// HandPanel.java
// Andrew Davison, July 2013, ad@fivedots.psu.ac.th

/* This panel repeatedly snaps a picture and draw it onto
   the panel. OpenCV is used, via the HandDetector class, to detect
   the user's gloved hand and label the fingers.

*/

import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.videoInputLib.*;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.Frame;

import java.awt.*;
import javax.swing.*;


public class HandPanel extends JPanel implements Runnable {
    /* dimensions of each image; the panel is the same size as the image */
    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;

    private static final int DELAY = 200;  // time (ms) between redraws of the panel

    private static final int CAMERA_ID = 1;


    private IplImage snapIm = null;
    private volatile boolean isRunning;
    private volatile boolean isFinished;

    // used for the average ms snap time information
    private int imageCount = 0;
    private long totalTime = 0;
    private Font msgFont;

    private HandDetector detector = null;   // for detecting hand and fingers


    public HandPanel() {
        setBackground(Color.white);
        msgFont = new Font("SansSerif", Font.BOLD, 18);

        new Thread(this).start();   // start updating the panel's image
    } // end of HandPanel()


    public Dimension getPreferredSize() {
        // make the panel wide enough for an image

        return new Dimension(WIDTH, HEIGHT);
    }


    public void run() {
        /* display the current webcam image every DELAY ms.
        Find the coloured rectangles in the image using HandDetector objects.
        The time statistics gathered here include the time taken to detect movement. */

        FrameGrabber grabber = initGrabber(CAMERA_ID);
        if (grabber == null) {
            return;
        }

        detector = new HandDetector("gloveHSV.txt", WIDTH, HEIGHT);
        // include the HSV color info about the user's gloved hand

        long duration;
        isRunning = true;
        isFinished = false;

        while (isRunning) {
            long startTime = System.currentTimeMillis();

            snapIm = picGrab(grabber, CAMERA_ID);
            imageCount++;
            detector.update(snapIm);
            repaint();

            duration = System.currentTimeMillis() - startTime;
            totalTime += duration;
            if (duration < DELAY) {
                try {
                    Thread.sleep(DELAY - duration);  // wait until DELAY time has passed
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
        closeGrabber(grabber, CAMERA_ID);
        System.out.println("Execution terminated");
        isFinished = true;
    }  // end of run()


    private FrameGrabber initGrabber(int ID) {
        FrameGrabber grabber = null;
        System.out.println("Initializing grabber for " + videoInput.getDeviceName(ID) + " ...");
        try {
            grabber = FrameGrabber.createDefault(ID);
            grabber.setFormat("dshow");       // using DirectShow
            grabber.setImageWidth(WIDTH);     // default is too small: 320x240
            grabber.setImageHeight(HEIGHT);
            grabber.start();
        } catch (Exception e) {
            System.out.println("Could not start grabber");
            System.out.println(e.getMessage());
            System.exit(1);
        }
        return grabber;
    }  // end of initGrabber()


    private IplImage picGrab(FrameGrabber grabber, int ID) {
        IplImage im = null;
        try {
            OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
            im = converter.convert(grabber.grabFrame());
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Problem grabbing image for camera " + ID);
        }
        return im;
    }  // end of picGrab()


    private void closeGrabber(FrameGrabber grabber, int ID) {
        try {
            grabber.stop();
            grabber.release();
        } catch (Exception e) {
            System.out.println("Problem stopping grabbing for camera " + ID);
        }
    }  // end of closeGrabber()


    public void paintComponent(Graphics g) {
      /* Draw the image, the detected hand and finger info, and the
      average ms snap time at the bottom left of the panel. */
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        Frame frame = grabberConverter.convert(snapIm);

        if (snapIm != null) {
            g2d.drawImage(paintConverter.getBufferedImage(frame, 1), 0, 0, this);
        }

        if (detector != null) {
            detector.draw(g2d);    // draws detected hand and finger info
        }

        writeStats(g2d);
    } // end of paintComponent()


    private void writeStats(Graphics2D g2d) {
        // write statistics in bottom-left corner, or "Loading" at start time
        g2d.setColor(Color.BLUE);
        g2d.setFont(msgFont);
        if (imageCount > 0) {
            String statsMsg = String.format("Snap Avg. Time:  %.1f ms",
                    ((double) totalTime / imageCount));
            g2d.drawString(statsMsg, 5, HEIGHT - 10);
            // write statistics in bottom-left corner
        } else { // no image yet
            g2d.drawString("Loading...", 5, HEIGHT - 10);
        }
    }  // end of writeStats()


    // --------------- called from the top-level JFrame ------------------

    public void closeDown() {
         /* Terminate run() and wait for it to finish.
         This stops the application from exiting until everything has finished. */
        isRunning = false;
        while (!isFinished) {
            try {
                Thread.sleep(DELAY);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    } // end of closeDown()

} // end of HandPanel class


