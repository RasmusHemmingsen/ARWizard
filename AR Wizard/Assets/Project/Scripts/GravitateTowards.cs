﻿using Leap.Unity.Interaction;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GravitateTowards : MonoBehaviour {
    [SerializeField]
    private string targetInHieracy;
    [SerializeField]
    private float maxSpeed;

    private GameObject targetHand;
    private Rigidbody rigidBody;

    void Start() {
        targetHand = GameObject.Find(targetInHieracy);
        rigidBody = GetComponent<Rigidbody>();

        // Get interaction manager from hierachy
        GetComponent<InteractionBehaviour>().manager = GameObject.Find("Interaction Manager").GetComponent<InteractionManager>();
    }

    void Update() {

        if (targetHand == null) {
            targetHand = GameObject.Find(targetInHieracy);
        }
        else if (targetHand.activeInHierarchy) {
            ApplyGravityTowardsTarget();
        }
    }

    private void ApplyGravityTowardsTarget() {
        Vector3 direction = targetHand.transform.position - transform.position;
        
        rigidBody.AddForce(direction.normalized * (direction.magnitude* direction.magnitude)*100);

        if(rigidBody.velocity.magnitude > maxSpeed) {
            rigidBody.velocity = rigidBody.velocity.normalized * maxSpeed;
        }
    }
}
