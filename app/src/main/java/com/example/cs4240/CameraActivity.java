package com.example.cs4240;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.firebase.ml.common.FirebaseMLException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class CameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    /*
    private CameraManager cameraManager;
    int cameraFacing;
    private TextureView.SurfaceTextureListener surfaceTextureListener;
    private String cameraId;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private CameraDevice.StateCallback stateCallback;
    private CameraCaptureSession cameraCaptureSession;
    private CameraDevice cameraDevice;
    private Size previewSize;
    private CaptureRequest.Builder captureRequestBuilder;
    private CaptureRequest captureRequest;
    private ImageReader imageReader;

    private TextureView textureView;

    private static final int CAMERA_REQUEST_CODE = 1888;

    private HandClassifier classifier;
    private Bitmap newBmp;
    */

    private static final String TAG = "test";
    private static final int LEFT = 200;
    private static final int RIGHT = 520;
    private static final int TOP = 150;
    private static final int BOTTOM = 450;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean  mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;
    private Bitmap bmp;
    private Bitmap overlayBmp;
    private boolean initBmp = false;
    private static final int CAMERA_PERMISSION = 555;
    private SignClassifier signClassifier;

    private CameraCharacteristics cameraCharacteristics;
    private CameraManager manager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera);

        //previewCameraSizes();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        }

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.cv_camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMinimumWidth(720); //doesn't seem to work
        mOpenCvCameraView.setMinimumHeight(1280);
        mOpenCvCameraView.setMaxFrameSize(720,1280);

        DisplayMetrics displayMetrics = getResources().getDisplayMetrics();
        float dpHeight = displayMetrics.heightPixels / displayMetrics.density;
        float dpWidth = displayMetrics.widthPixels / displayMetrics.density;
        float height = (dpWidth *1280)/720;
        overlayBmp = Bitmap.createBitmap(720,1280, Bitmap.Config.ARGB_8888);
        overlayBmp = Bitmap.createScaledBitmap(overlayBmp, 720 , (int)height, true);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.FILL);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(10.0f);
        paint.setColor(Color.RED);
        Canvas canvas = new Canvas(overlayBmp);
        canvas.drawRect(LEFT, TOP, RIGHT, BOTTOM, paint);

        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageBitmap(overlayBmp);

        try {
            signClassifier = new SignClassifier(this, "resnet50_9000.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (!initBmp) {
            bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
            initBmp = true;
        }
        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 720,1280, 0);
        Core.flip(mRgbaF, mRgba, 1 );

        Utils.matToBitmap(mRgba, bmp);

        predictSign(LEFT, TOP, RIGHT-LEFT, BOTTOM-TOP);

        //overlayBmp.eraseColor(Color.TRANSPARENT);
//        try {
//            handClassifier.predict(bmp);
//        } catch (FirebaseMLException e) {
//            e.printStackTrace();
//        }

        return mRgba; // This function must return
    }

    public Bitmap getBitmap() {
        return bmp;
    }

    public void predictSign(int x, int y, int w, int h){

        if (w <= 0 || h <= 0) {
            return;
        }

        if (x <= 0 || y <= 0) {
            return;
        }

        Log.d("parameters", x + " : " + y + " : " + w + " : " + h);
        Bitmap roi = Bitmap.createBitmap(bmp, x, y, w, h);

        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageBitmap(roi);

        try {
            signClassifier.predict(roi);
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    public void setTextImage() {
        this.signClassifier.label(overlayBmp);
        ImageView imageView = (ImageView)findViewById(R.id.im_view);
        imageView.setImageBitmap(overlayBmp);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,  String permissions[], int[] grantResults) {
        switch (requestCode) {
            case CAMERA_PERMISSION:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //if(mClss != null) {
                        Intent intent = new Intent(this, CameraActivity.class);
                        startActivity(intent);
                    //}
                } else {
                    Toast.makeText(this, "Please grant camera permission to use the QR Scanner", Toast.LENGTH_SHORT).show();
                }
                return;
        }
    }

    private void previewCameraSizes() {
        // Check camera supported sizes
        manager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraCharacteristics = manager.getCameraCharacteristics(manager.getCameraIdList()[0]);
            StreamConfigurationMap scmap = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] previewSizes = scmap.getOutputSizes(ImageReader.class);
            for (int i = 0; i < previewSizes.length; i++) {
                //Log.d("test", previewSizes[i].toString());
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
}
