using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Leap;
using LeapInternal;
using Newtonsoft.Json;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class HandThroughLeap : MonoBehaviour
{
    public bool TrainData = false;
    public string TrainGestureName;

    private Controller _controller;
    private Frame _frame;
    private readonly List<string> _dataDescription = new List<string>();
    private readonly List<float> _data = new List<float>();
    private static bool _onlyOnce;
    private const int GESTURE_ITERATION = 30;
    private int _counterFrameRate;
    private int _counterWriteToCsv;
    private HttpClient _client;

    public delegate void PythonEvent(Gesture g);
    public static event PythonEvent HandGesturePercentageEvent;



    // Start is called before the first frame update
    void Start()
    {
        _controller = new Controller();
        _client = new HttpClient();
        _client.BaseAddress = new Uri("http://localhost:5000");
    }

    // Update is called once per frame
    async void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            HandGesturePercentageEvent(new Gesture("0", 100f));
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            HandGesturePercentageEvent(new Gesture("2", 100f));
        }

        _frame = _controller.Frame();
        if (_counterFrameRate++ >= 5)
        {
            if (_frame.Hands.FirstOrDefault() == null) return;

            InitCsv(_frame.Hands.First());
            ExtractHandData(_frame.Hands.First());
            _counterFrameRate = 0;
            if (++_counterWriteToCsv >= GESTURE_ITERATION)
            {
                if (TrainData)
                {
                    WriteDataToCsv();
                }
                var json = JsonConvert.SerializeObject(_data);
                var content = new StringContent(json);
                _data.RemoveRange(0, _data.Count / 3);
                _counterWriteToCsv = 20;

                if (TrainData) return;

                var response = await _client.PostAsync("/postjson", content);
                var result = await response.Content.ReadAsStringAsync();

                // raise event with result
                Debug.Log(result);
                HandGesturePercentageEvent(new Gesture(result, 100f));
            }
        }
    }


    private void ExtractHandData(Hand hand)
    {
        //Palm
        _data.Add(hand.PalmPosition.x);
        _data.Add(hand.PalmPosition.y);
        _data.Add(hand.PalmPosition.z);
        _data.Add(hand.PalmPosition.Yaw);
        _data.Add(hand.PalmPosition.Pitch);
        _data.Add(hand.PalmPosition.Roll);

        //Wrist
        _data.Add(hand.WristPosition.x);
        _data.Add(hand.WristPosition.y);
        _data.Add(hand.WristPosition.z);
        _data.Add(hand.WristPosition.Yaw);
        _data.Add(hand.WristPosition.Pitch);
        _data.Add(hand.WristPosition.Roll);



        //Fingers
        foreach (var handFinger in hand.Fingers)
        {
            foreach (var bone in handFinger.bones)
            {
                _data.Add(bone.NextJoint.x);
                _data.Add(bone.NextJoint.y);
                _data.Add(bone.NextJoint.z);
                _data.Add(bone.NextJoint.Yaw);
                _data.Add(bone.NextJoint.Pitch);
                _data.Add(bone.NextJoint.Roll);
                _data.Add(hand.PalmPosition.DistanceTo(bone.NextJoint));
            }
        }
    }

    private void InitCsv(Hand hand)
    {
        if (_onlyOnce) return;
        _onlyOnce = true;

        Debug.Log("Writing Description to file");
        for (int i = 0; i < GESTURE_ITERATION; i++)
        {
            _dataDescription.Add($"{i}. Palm Position X");
            _dataDescription.Add($"{i}. Palm Position Y");
            _dataDescription.Add($"{i}. Palm Position Z");
            _dataDescription.Add($"{i}. Palm Position Yaw");
            _dataDescription.Add($"{i}. Palm Position Pitch");
            _dataDescription.Add($"{i}. Palm Position Roll");
            _dataDescription.Add($"{i}. Wrist Position X");
            _dataDescription.Add($"{i}. Wrist Position Y");
            _dataDescription.Add($"{i}. Wrist Position Z");
            _dataDescription.Add($"{i}. Wrist Position Yaw");
            _dataDescription.Add($"{i}. Wrist Position Pitch");
            _dataDescription.Add($"{i}. Wrist Position Roll");

            foreach (var handFinger in hand.Fingers)
            {
                foreach (var bone in handFinger.bones)
                {
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} X");
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} Y");
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} Z");
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} Yaw");
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} Pitch");
                    _dataDescription.Add($"{i}. Finger: {handFinger.Id} Bone: {bone.Type} Roll");
                    _dataDescription.Add($"{i}. Palm Dist To Finger: {handFinger.Id} Bone: {bone.Type}");
                }
            }
        }

        File.AppendAllLines($"Assets\\TrainingData\\{TrainGestureName}.csv", new[] { string.Join(",", _dataDescription) });
    }

    private void WriteDataToCsv()
    {
        Debug.Log("Writing Data to file");
        File.AppendAllLines($"Assets\\TrainingData\\{TrainGestureName}.csv", new[] { string.Join(",", _data) });
    }
}
