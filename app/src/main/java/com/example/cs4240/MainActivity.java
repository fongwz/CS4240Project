package com.example.cs4240;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

public class MainActivity extends AppCompatActivity {

    private static final int test_img = R.drawable.test_img;
    private static final String model_file = "hand_landmark.tflite";

    private HandClassifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            Bitmap bmp = BitmapFactory.decodeResource(this.getResources(), test_img);
            Bitmap newBmp = bmp.copy(android.graphics.Bitmap.Config.ARGB_8888, true);

            //classifier = new HandClassifier(this, model_file, 42);
            //classifier.predict(bmp);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setImage() {
        Bitmap bmp = BitmapFactory.decodeResource(this.getResources(), test_img);
        Bitmap newBmp = bmp.copy(android.graphics.Bitmap.Config.ARGB_8888, true);
        this.classifier.label(newBmp);
        //ImageView imView = (ImageView)findViewById(R.id.im_view);
        //imView.setImageBitmap(newBmp);
    }

    public void onButtonClick(View view) {
        Intent i = new Intent(this, CameraActivity.class);
        startActivity(i);
    }
}
