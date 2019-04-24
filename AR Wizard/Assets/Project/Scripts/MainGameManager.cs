using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MainGameManager : MonoBehaviour
{
    private void Awake()
    {
        HandThroughLeap.HandGesturePercentageEvent += OnEvent;
    }

    private void OnEvent(string s)
    {

    }
}
